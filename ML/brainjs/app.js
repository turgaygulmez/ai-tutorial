/**
 * Possibilities are infinite
 * - Task: train your network with job groups and their related lessons
 *    - Ask user to insert what subjects they enjoy most, output what is the potential jobs they can pick up in the future
 *    - Ex:
 *        --- math, physics, science, english, history, geography
 *        lessons: [true, true, true, false, false, false],
 *        job: { engineering: 1 }
 */

const network = new brain.NeuralNetwork();

// define your training data
// index mapping for skills
// [canSwim, canRun, hasFin, hasLegs, hasTail, canFly]

const animals = [
  {
    skills: [true, true, false, true, true, false],
    type: { dog: 1 },
  },
  {
    skills: [true, false, true, false, false, false],
    type: { fish: 1 },
  },
  {
    skills: [false, false, false, true, true, true],
    type: { bird: 1 },
  },
];

// Train the Network with 4 input objects
network.train(
  animals.map((animal) => {
    return {
      input: animal.skills,
      output: animal.type,
    };
  })
);

// what skills you want in your pet?
const canSwim = true,
  canRun = false,
  hasFin = true,
  hasLegs = false,
  hasTail = false,
  canFly = false;

let result = network.run([canSwim, canRun, hasFin, hasLegs, hasTail, canFly]);

console.log(result);

console.log(
  Object.keys(result).reduce((a, b) => (result[a] > result[b] ? a : b))
);
