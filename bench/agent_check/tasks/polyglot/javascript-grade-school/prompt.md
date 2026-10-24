# Grade School

# Instructions

Given students' names along with the grade they are in, create a roster for the school.

In the end, you should be able to:

- Add a student's name to the roster for a grade:
  - "Add Jim to grade 2."
  - "OK."
- Get a list of all students enrolled in a grade:
  - "Which students are in grade 2?"
  - "We've only got Jim right now."
- Get a sorted list of all students in all grades.
  Grades should be sorted as 1, 2, 3, etc., and students within a grade should be sorted alphabetically by name.
  - "Who is enrolled in school right now?"
  - "Let me think.
    We have Anna, Barb, and Charlie in grade 1, Alex, Peter, and Zoe in grade 2, and Jim in grade 5.
    So the answer is: Anna, Barb, Charlie, Alex, Peter, Zoe, and Jim."

Note that all our students only have one name (it's a small town, what do you want?), and each student cannot be added more than once to a grade or the roster.
If a test attempts to add the same student more than once, your implementation should indicate that this is incorrect.

---

Implement this exercise in `grade-school.js`, the stub in the current directory. Keep the names of the existing functions and
classes, since the tests import them. Use only the JavaScript standard library; no packages can be installed.

`grade-school.spec.js` holds some of the tests; run them with `npm test`. A hidden test suite with more cases will be run on
your solution when you finish.
