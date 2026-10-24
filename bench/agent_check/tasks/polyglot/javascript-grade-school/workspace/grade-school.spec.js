import { beforeEach, describe, expect, test, xtest } from '@jest/globals';
import { GradeSchool } from './grade-school';

describe('Grade School', () => {
  let school;

  beforeEach(() => {
    school = new GradeSchool();
  });

  test('Roster is empty when no student is added', () => {
    expect(school.roster()).toEqual([]);
  });

  test('Adding multiple students in the same grade in the roster', () => {
    expect(school.add('Blair', 2)).toEqual(true);
    expect(school.add('James', 2)).toEqual(true);
    expect(school.add('Paul', 2)).toEqual(true);
  });

  test('Student not added to same grade in the roster more than once', () => {
    school.add('Blair', 2);
    school.add('James', 2);
    school.add('James', 2);
    school.add('Paul', 2);

    const expectedDb = ['Blair', 'James', 'Paul'];
    expect(school.roster()).toEqual(expectedDb);
  });

  test('Cannot add same student to multiple grades in the roster', () => {
    expect(school.add('Blair', 2)).toEqual(true);
    expect(school.add('James', 2)).toEqual(true);
    expect(school.add('James', 3)).toEqual(false);
    expect(school.add('Paul', 3)).toEqual(true);
  });

  test('Students are sorted by name in the roster', () => {
    school.add('Peter', 2);
    school.add('Zoe', 2);
    school.add('Alex', 2);

    const expectedDb = ['Alex', 'Peter', 'Zoe'];
    expect(school.roster()).toEqual(expectedDb);
  });

  test('Grade is empty if no students in that grade', () => {
    school.add('Peter', 2);
    school.add('Zoe', 2);
    school.add('Alex', 2);
    school.add('Jim', 3);

    expect(school.grade(1)).toEqual([]);
  });

  test('Student not added to other grade for multiple grades', () => {
    school.add('Blair', 2);
    school.add('James', 2);
    school.add('James', 3);
    school.add('Paul', 3);

    const expectedDb = ['Paul'];
    expect(school.grade(3)).toEqual(expectedDb);
  });
});
