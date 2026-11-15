// Exercise playback and decision labels with a minimal DOM and timer harness.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
class Element {
  constructor() { this.children = []; this.events = {}; this.value = '0'; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  add(option) { this.children.push(option); }
  addEventListener(name, fn) { this.events[name] = fn; }
  emit(name) { this.events[name](); }
}
const ids = ['data', 'track', 'seek', 'play', 'field', 'source', 'frame', 'summary',
  'directions', 'layers', 'previous', 'next', 'trace-legend', 'branch-legend'];
const elements = Object.fromEntries(ids.map(id => [id, new Element()]));
elements.data.textContent = html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1];
elements.field.width = 640;
elements.field.height = 520;
const arcs = [];
elements.field.getContext = () => new Proxy({}, {get: (_, key) => (...args) => {
  if (key === 'clearRect') arcs.length = 0;
  if (key === 'arc') arcs.push(args);
}});
for (const id of ['track', 'seek', 'play', 'previous', 'next']) {
  elements[id].disabled = new RegExp(`<[^>]+id="${id}"[^>]* disabled`).test(html);
}
let interval;
vm.runInNewContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], {
  document: {getElementById: id => elements[id], createElement: () => new Element(), addEventListener: () => {}},
  Option: function (text, value) { this.text = text; this.value = value; },
  setInterval: fn => { interval = fn; return 1; },
  clearInterval: () => { interval = null; },
});
const {directions, frame, next, play, seek, summary, track} = elements;
if (elements.data.textContent.trim().startsWith('__REPLAY_')) {
  assert.match(elements.source.textContent, /Open the generated replay\.html/);
  assert(track.disabled && seek.disabled && play.disabled && next.disabled && elements.previous.disabled);
  assert.equal(track.children.length, 0);
  console.log('Replay template shows its empty-state instructions.');
  process.exit(0);
}
assert.equal(directions.children.length, 9);
assert.equal(track.disabled, false);
const initialTrack = JSON.parse(elements.data.textContent)[0];
if (initialTrack.counterfactual_window) {
  assert.equal(elements['branch-legend'].hidden, false);
  assert(summary.children.some(item => item.textContent === 'Branch action'));
  next.emit('click');
  const selected = frame.textContent.split(' · ')[0];
  track.value = '1';
  track.emit('change');
  assert.match(elements.source.textContent, /refreshed recorded hazards/);
  assert.equal(frame.textContent.split(' · ')[0], selected);
  assert(summary.children.some(item => item.textContent === 'Decision source'));
  console.log('Counterfactual replay shows branch labels and aligned frames.');
  process.exit(0);
}
if (initialTrack.trace_window) {
  assert.match(elements.source.textContent, /offline predictions/);
  assert.equal(elements['trace-legend'].hidden, false);
  assert(summary.children.some(item => item.textContent === 'Recorded game action'));
  next.emit('click');
  const sample = initialTrack.frames[1];
  assert(frame.textContent.startsWith(`Frame ${sample.frame} `));
  const [xmin, xmax, ymin, ymax] = sample.scene.bounds;
  const scale = Math.min(600 / (xmax - xmin), 480 / (ymax - ymin));
  const circle = sample.scene.circles[0];
  assert.equal(arcs[0][0], (640 - (xmax - xmin) * scale) / 2 + (circle.x - xmin) * scale);
  assert.equal(arcs[0][1], (520 - (ymax - ymin) * scale) / 2 + (circle.y - ymin) * scale);
  track.value = '1';
  track.emit('change');
  assert.equal(Number(seek.value), 1);
  seek.value = seek.max;
  seek.emit('input');
  assert.equal(summary.children.some(item => item.textContent === 'Recorded death first observed'),
    initialTrack.frames.at(-1).death_observed != null);
  console.log('Trace replay preserves decision alignment and renders recorded geometry.');
  process.exit(0);
}
assert.match(frame.textContent, /Frame 0/);
next.emit('click');
assert.match(frame.textContent, /Frame 4/);
play.emit('click');
assert.equal(play.textContent, 'Pause');
interval();
assert.match(frame.textContent, /Frame 8/);
play.emit('click');
assert.equal(interval, null);
seek.value = 3;
seek.emit('input');
assert.match(frame.textContent, /Frame 12/);
function select(label) {
  const option = track.children.find(item => item.text === label);
  if (!option) return false;
  track.value = option.value;
  track.emit('change');
  return true;
}
if (select('Time cutoff / checked prefix')) {
  assert(summary.children.some(item => item.textContent === 'Time budget reached'));
  assert(play.disabled && seek.disabled);
  assert(select('detour_before_approach / short_ttc'));
  assert.match(elements.source.textContent, /Missed objective/);
  assert(select('detour_before_approach / static_guided_ttc'));
  assert.match(elements.source.textContent, /Goal reached at frame 28/);
  assert(select('evacuate_before_short_horizon_sees_laser / short_clearance'));
  seek.value = seek.max;
  seek.emit('input');
  assert(summary.children.some(item => item.textContent === 'All extensions blocked by collision or bounds'));
} else {
  assert(summary.children.some(item => item.textContent === 'Manager option'));
  assert(directions.children.every(card => card.children.some(item => item.textContent.endsWith('%'))));
}
console.log('Replay controls and decision labels passed.');
