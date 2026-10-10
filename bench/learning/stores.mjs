// Build each pull request's time-correct lesson store with Rigour's own learner, then list the lessons the
// reviewer would be served on each head (docs/LEARNING.md). No model, no GitHub API: the learner reads the
// crawl through an injected fetch that serves only what existed before the cutoff. Git runs in the clone
// (the learner may fetch a pull request's head ref to read the commits a comment was written on).
//
// node stores.mjs <core dir> <crawl.json> <clone dir> <heads.json> <out dir>
import fs from 'fs';
import path from 'path';
import { pathToFileURL } from 'url';

const API = 'https://cached.invalid';
const MODES = ['verified', 'all'];

/** The judge's lesson limits, read from the installed reviewer so the pre-check serves exactly what it would. */
export function judgeLimits(source) {
  const value = (name) => {
    const match = new RegExp(`const ${name} = (\\d+);`).exec(source);
    if (!match) throw new Error(`${name} not found in the reviewer's context.js: the pre-check cannot match this version`);
    return Number(match[1]);
  };
  const call = /lessonsForDiff\(input\.cwd, input\.diff, input\.lessons, JUDGE_STANDARDS, JUDGE_FILE_LESSONS, JUDGE_LESSONS_PER_FILE/;
  if (!call.test(source)) throw new Error('the reviewer no longer calls lessonsForDiff as the pre-check does');
  return { standards: value('JUDGE_STANDARDS'), limit: value('JUDGE_FILE_LESSONS'), perFile: value('JUDGE_LESSONS_PER_FILE') };
}

/** What GitHub would have answered before `cutoff`: merged pull requests, their comments and reviews. */
export function cachedFetch(crawl, cutoff) {
  const base = `${API}/repos/${crawl.repo}`;
  const listing = crawl.prs.filter(pr => pr.merged_at < cutoff).sort((a, b) => b.merged_at.localeCompare(a.merged_at));
  const before = (items, key) => items.filter(i => i[key] && i[key] < cutoff).sort((a, b) => a[key].localeCompare(b[key]));
  const answer = (data) => ({ ok: true, status: 200, json: async () => data });
  return async (url) => {
    const u = new URL(url);
    const rest = `${u.origin}${u.pathname}`.slice(base.length);
    if (!`${u.origin}${u.pathname}`.startsWith(base)) throw new Error(`unexpected request ${url}`);
    const size = Number(u.searchParams.get('per_page') ?? '30');
    const page = (items) => {
      const n = Number(u.searchParams.get('page') ?? '1');
      return answer(items.slice((n - 1) * size, n * size));
    };
    if (rest === '/pulls') return page(listing);
    const match = /^\/pulls\/(\d+)\/(comments|reviews)$/.exec(rest);
    const entry = match && crawl.reviews[match[1]];
    if (!entry) throw new Error(`unexpected request ${url}`);
    return page(match[2] === 'comments' ? before(entry.comments, 'created_at') : before(entry.reviews, 'submitted_at'));
  };
}

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (err) {
    throw new Error(`unreadable ${file}: ${err.message}`);
  }
}

function served(core, clone, diff, limits) {
  return Object.fromEntries(MODES.map(mode => [mode,
    core.lessonsForDiff(clone, diff, mode, limits.standards, limits.limit, limits.perFile).map(l => l.id)]));
}

async function buildStore(core, crawl, clone, pr, outDir, limits) {
  if (!pr.main_ref) throw new Error(`pull request ${pr.pr} has no main_ref: run bench learning prepare`);
  const file = path.join(outDir, 'stores', `${pr.pr}.json`);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.rmSync(file, { force: true });
  process.env.RIGOUR_REVIEW_LESSONS = file;
  const learned = await core.learnFromReviews(clone, {
    fetch: cachedFetch(crawl, pr.cutoff), token: 'cached', repo: crawl.repo, apiUrl: API, until: pr.cutoff,
    limit: crawl.limit, mainRef: pr.main_ref,
  });
  const heads = Object.fromEntries(pr.heads.map(h => [h.sha, h.error ? { error: h.error }
    : served(core, clone, fs.readFileSync(h.diff, 'utf8'), limits)]));
  return { pr: pr.pr, cutoff: pr.cutoff, main_ref: pr.main_ref, store: path.relative(outDir, file), learned, heads };
}

async function main([coreDir, crawlFile, clone, headsFile, outDir]) {
  const core = await import(pathToFileURL(path.join(coreDir, 'dist', 'index.js')).href);
  const limits = judgeLimits(fs.readFileSync(path.join(coreDir, 'dist', 'review', 'reviewer', 'context.js'), 'utf8'));
  const crawl = readJson(crawlFile);
  const prs = readJson(headsFile);
  const results = [];
  for (const pr of prs) results.push(await buildStore(core, crawl, clone, pr, outDir, limits));
  const version = readJson(path.join(coreDir, 'package.json')).version;
  fs.writeFileSync(path.join(outDir, 'served.json'), JSON.stringify({ repo: crawl.repo, core: version, limits, prs: results }, null, 1));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main(process.argv.slice(2)).catch(err => { console.error(err.message); process.exit(1); });
}
