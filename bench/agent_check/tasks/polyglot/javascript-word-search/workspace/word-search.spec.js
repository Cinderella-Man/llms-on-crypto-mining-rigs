import { describe, expect, test, xtest } from '@jest/globals';
import WordSearch from './word-search';

describe('Word Search', () => {
  test('Should accept an initial game grid', () => {
    const grid = ['jefblpepre'];
    const wordSearch = new WordSearch(grid);

    expect(wordSearch instanceof WordSearch).toEqual(true);
  });

  test('Can locate a left to right word in a different position', () => {
    const grid = ['mtclojurer'];
    const expectedResults = {
      clojure: {
        start: [1, 3],
        end: [1, 9],
      },
    };

    const wordSearch = new WordSearch(grid);

    expect(wordSearch.find(['clojure'])).toEqual(expectedResults);
  });

  test('Can locate a left to right word in a two line grid', () => {
    const grid = ['jefblpepre', 'clojurermt'];

    const expectedResults = {
      clojure: {
        start: [2, 1],
        end: [2, 7],
      },
    };

    const wordSearch = new WordSearch(grid);

    expect(wordSearch.find(['clojure'])).toEqual(expectedResults);
  });

  test('Can locate a left to right word in a ten line grid', () => {
    const grid = [
      'jefblpepre',
      'camdcimgtc',
      'oivokprjsm',
      'pbwasqroua',
      'rixilelhrs',
      'wolcqlirpc',
      'screeaumgr',
      'alxhpburyi',
      'jalaycalmp',
      'clojurermt',
    ];

    const expectedResults = {
      clojure: {
        start: [10, 1],
        end: [10, 7],
      },
    };

    const wordSearch = new WordSearch(grid);

    expect(wordSearch.find(['clojure'])).toEqual(expectedResults);
  });

  test('Can find two words written left to right', () => {
    const grid = [
      'aefblpepre',
      'camdcimgtc',
      'oivokprjsm',
      'pbwasqroua',
      'rixilelhrs',
      'wolcqlirpc',
      'screeaumgr',
      'alxhpburyi',
      'jalaycalmp',
      'clojurermt',
      'xjavamtzlp',
    ];

    const expectedResults = {
      clojure: {
        start: [10, 1],
        end: [10, 7],
      },
      java: {
        start: [11, 2],
        end: [11, 5],
      },
    };

    const wordSearch = new WordSearch(grid);

    expect(wordSearch.find(['java', 'clojure'])).toEqual(expectedResults);
  });

  test('Should locate words written top to bottom', () => {
    const grid = [
      'jefblpepre',
      'camdcimgtc',
      'oivokprjsm',
      'pbwasqroua',
      'rixilelhrs',
      'wolcqlirpc',
      'screeaumgr',
      'alxhpburyi',
      'jalaycalmp',
      'clojurermt',
    ];

    const expectedResults = {
      clojure: {
        start: [10, 1],
        end: [10, 7],
      },
      elixir: {
        start: [5, 6],
        end: [5, 1],
      },
      ecmascript: {
        start: [1, 10],
        end: [10, 10],
      },
    };

    const wordSearch = new WordSearch(grid);

    expect(wordSearch.find(['elixir', 'clojure', 'ecmascript'])).toEqual(
      expectedResults,
    );
  });

  test('Should locate words written bottom right to top left', () => {
    const grid = [
      'jefblpepre',
      'camdcimgtc',
      'oivokprjsm',
      'pbwasqroua',
      'rixilelhrs',
      'wolcqlirpc',
      'screeaumgr',
      'alxhpburyi',
      'jalaycalmp',
      'clojurermt',
    ];

    const expectedResults = {
      clojure: {
        start: [10, 1],
        end: [10, 7],
      },
      elixir: {
        start: [5, 6],
        end: [5, 1],
      },
      ecmascript: {
        start: [1, 10],
        end: [10, 10],
      },
      rust: {
        start: [5, 9],
        end: [2, 9],
      },
      java: {
        start: [1, 1],
        end: [4, 4],
      },
      lua: {
        start: [9, 8],
        end: [7, 6],
      },
    };

    const wordSearch = new WordSearch(grid);

    expect(
      wordSearch.find([
        'clojure',
        'elixir',
        'ecmascript',
        'rust',
        'java',
        'lua',
      ]),
    ).toEqual(expectedResults);
  });

  test('Should fail to locate a word that is not in the puzzle', () => {
    const grid = [
      'jefblpepre',
      'camdcimgtc',
      'oivokprjsm',
      'pbwasqroua',
      'rixilelhrs',
      'wolcqlirpc',
      'screeaumgr',
      'alxhpburyi',
      'jalaycalmp',
      'clojurermt',
    ];

    const expectedResults = {
      clojure: {
        start: [10, 1],
        end: [10, 7],
      },
      elixir: {
        start: [5, 6],
        end: [5, 1],
      },
      ecmascript: {
        start: [1, 10],
        end: [10, 10],
      },
      rust: {
        start: [5, 9],
        end: [2, 9],
      },
      java: {
        start: [1, 1],
        end: [4, 4],
      },
      lua: {
        start: [9, 8],
        end: [7, 6],
      },
      lisp: {
        start: [6, 3],
        end: [3, 6],
      },
      ruby: {
        start: [6, 8],
        end: [9, 5],
      },
      haskell: undefined,
    };

    const wordSearch = new WordSearch(grid);

    expect(
      wordSearch.find([
        'clojure',
        'elixir',
        'ecmascript',
        'rust',
        'java',
        'lua',
        'lisp',
        'ruby',
        'haskell',
      ]),
    ).toEqual(expectedResults);
  });

  test('Should not wrap around horizontally to find a word', () => {
    const grid = ['silabcdefp'];

    const expectedResults = {
      lisp: undefined,
    };

    const wordSearch = new WordSearch(grid);

    expect(wordSearch.find(['lisp'])).toEqual(expectedResults);
  });
});
