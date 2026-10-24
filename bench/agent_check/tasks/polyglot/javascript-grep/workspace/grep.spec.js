// @ts-check

const { test, xtest, describe, expect } = require('@jest/globals');

const { spawnSync } = require('child_process');
const { resolve: resolvePath, relative } = require('path');

const BASE_DIR = resolvePath(__dirname);

/**
 * Spawns a new node process, emulating `grep`.
 *
 * node /path/to/grep.js -f -l -a -g -s **pattern** files1 files2 files3
 *
 * @param {{ flags: string[], pattern: string, files: string[] }} config
 */
function spawnGrep(config) {
  const args = [
    resolvePath(BASE_DIR, 'grep.js'),
    ...config.flags,
    config.pattern,
    ...config.files.map((file) =>
      relative(BASE_DIR, resolvePath(BASE_DIR, 'data', file)),
    ),
  ];

  return new Promise((resolve, reject) => {
    const child = spawnSync('node', args, { stdio: 'pipe', cwd: BASE_DIR });
    const stderr = child.stderr.toString().trim();
    const stdout = child.stdout.toString().trim().split(/\r?\n/).join('\n');

    // If anything is written to stderr, consider the entire process as failed.
    //
    // Normally you'd check the status code (extest code) and reject/fail if it's
    // not equal to "0".
    if (stderr) {
      reject(stderr);
    } else {
      resolve(stdout);
    }
  });
}

/**
 * Formats a string template cross-env
 *
 * @param {string} stringTemplate
 */
function formatStringTemplate(stringTemplate) {
  return stringTemplate
    .split(/\r?\n/)
    .map((sentence) => sentence.trim())
    .join('\n');
}

/**
 * Path to a data file
 * @param {string} file
 */
function resolveDataFile(file) {
  return relative(BASE_DIR, resolvePath(BASE_DIR, 'data', file));
}

describe('grep exercise', () => {
  describe('Test grepping a single file', () => {
    test('One file, one match, no flags', () => {
      return expect(
        spawnGrep({
          pattern: 'Agamemnon',
          flags: [],
          files: ['iliad.txt'],
        }),
      ).resolves.toBe('Of Atreus, Agamemnon, King of men.');
    });

    test('One file, one match, print file names flag', () => {
      return expect(
        spawnGrep({
          pattern: 'Forbidden',
          flags: ['-l'],
          files: ['paradise-lost.txt'],
        }),
      ).resolves.toBe(resolveDataFile('paradise-lost.txt'));
    });

    test('One file, several matches, no flags', () => {
      return expect(
        spawnGrep({
          pattern: 'may',
          flags: [],
          files: ['midsummer-night.txt'],
        }),
      ).resolves.toBe(
        formatStringTemplate(`Nor how it may concern my modesty,
        But I beseech your grace that I may know
        The worst that may befall me in this case,`),
      );
    });

    test('One file, several matches, case-insensitive flag', () => {
      return expect(
        spawnGrep({
          pattern: 'ACHILLES',
          flags: ['-i'],
          files: ['iliad.txt'],
        }),
      ).resolves.toBe(
        formatStringTemplate(`Achilles sing, O Goddess! Peleus' son;
        The noble Chief Achilles from the son`),
      );
    });

    test('One file, one match, file flag takes precedence over line flag', () => {
      return expect(
        spawnGrep({
          pattern: 'ten',
          flags: ['-n', '-l'],
          files: ['iliad.txt'],
        }),
      ).resolves.toBe(resolveDataFile('iliad.txt'));
    });
  });

  describe('Test grepping multiples files at once', () => {
    test('Multiple files, one match, no flags', () => {
      return expect(
        spawnGrep({
          pattern: 'Agamemnon',
          flags: [],
          files: ['iliad.txt', 'midsummer-night.txt', 'paradise-lost.txt'],
        }),
      ).resolves.toBe(
        `${resolveDataFile('iliad.txt')}:Of Atreus, Agamemnon, King of men.`,
      );
    });

    test('Multiple files, one match, print file names flag', () => {
      return expect(
        spawnGrep({
          pattern: 'who',
          flags: ['-l'],
          files: ['iliad.txt', 'midsummer-night.txt', 'paradise-lost.txt'],
        }),
      ).resolves.toBe(
        formatStringTemplate(`${resolveDataFile('iliad.txt')}
        ${resolveDataFile('paradise-lost.txt')}`),
      );
    });

    test('Multiple files, one match, match entire lines flag', () => {
      return expect(
        spawnGrep({
          pattern: 'But I beseech your grace that I may know',
          flags: ['-x'],
          files: ['iliad.txt', 'midsummer-night.txt', 'paradise-lost.txt'],
        }),
      ).resolves.toBe(
        `${resolveDataFile(
          'midsummer-night.txt:But I beseech your grace that I may know',
        )}`,
      );
    });

    test('Multiple files, several matches, file flag takes precedence over line number flag', () => {
      return expect(
        spawnGrep({
          pattern: 'who',
          flags: ['-n', '-l'],
          files: ['iliad.txt', 'midsummer-night.txt', 'paradise-lost.txt'],
        }),
      ).resolves.toBe(
        formatStringTemplate(`${resolveDataFile('iliad.txt')}
        ${resolveDataFile('paradise-lost.txt')}`),
      );
    });
  });
});
