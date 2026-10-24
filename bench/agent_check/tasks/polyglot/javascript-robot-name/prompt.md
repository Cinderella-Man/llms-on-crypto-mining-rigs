# Robot Name

# Instructions

Manage robot factory settings.

When a robot comes off the factory floor, it has no name.

The first time you turn on a robot, a random name is generated in the format of two uppercase letters followed by three digits, such as RX837 or BC811.

Every once in a while we need to reset a robot to its factory settings, which means that its name gets wiped.
The next time you ask, that robot will respond with a new random name.

The names must be random: they should not follow a predictable sequence.
Using random names means a risk of collisions.
Your solution must ensure that every existing robot has a unique name.

---

Implement this exercise in `robot-name.js`, the stub in the current directory. Keep the names of the existing functions and
classes, since the tests import them. Use only the JavaScript standard library; no packages can be installed.

`robot-name.spec.js` holds some of the tests; run them with `npm test`. A hidden test suite with more cases will be run on
your solution when you finish.
