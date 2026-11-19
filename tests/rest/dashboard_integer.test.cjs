"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const { dashboard } = require("./dashboard_fixture.cjs");

test("a seed outside the safe integer range is not displayed or sent", async () => {
  const ui = dashboard(
    JSON.parse('{"type":"int","value":9007199254740993}'),
    "seed",
  );
  await ui.refresh();
  assert.equal(ui.value(), "");
  assert.match(ui.placeholder(), /safe integer range/);

  ui.edit("9007199254740993");
  await ui.setAll();
  assert.match(ui.status(), /^invalid value: .*safe integer range/);
  assert.equal(ui.commands.length, 0);

  ui.edit("1.00000000000000001");
  await ui.setAll();
  assert.match(ui.status(), /^invalid value: .*whole number/);
  assert.equal(ui.commands.length, 0);

  ui.edit("-9007199254740991");
  const submission = ui.setAll();
  assert.equal(ui.commands.length, 1);
  assert.equal(ui.commands[0].value, -9007199254740991);
  ui.complete(0);
  await submission;
  assert.equal(ui.status(), "updated");
});

test("integer tensor edits reject unsafe nested elements before sending", async () => {
  const ui = dashboard(
    JSON.parse(
      '{"type":"tensor","dtype":"torch.int64","shape":[1,2],"value":[[0,-9007199254740993]]}',
    ),
    "indices",
  );
  await ui.refresh();
  assert.equal(ui.value(), "");
  assert.match(ui.placeholder(), /safe integer range/);

  ui.edit("[[0,-9007199254740993]]");
  await ui.setAll();
  assert.match(ui.status(), /^invalid value: .*safe integer range/);
  assert.equal(ui.commands.length, 0);

  ui.edit("[[0,1.00000000000000001]]");
  await ui.setAll();
  assert.match(ui.status(), /^invalid value: .*whole numbers/);
  assert.equal(ui.commands.length, 0);

  ui.edit("[[9007199254740991,-7]]");
  const submission = ui.setAll();
  assert.equal(ui.commands.length, 1);
  assert.deepEqual(ui.commands[0].value, [[9007199254740991, -7]]);
  ui.complete(0);
  await submission;
  assert.equal(ui.status(), "updated");
});

test("floating-point and boolean tunables keep their existing input behavior", async () => {
  const cases = [
    [{ type: "float", value: 1e18 }, "-1.5e18", -1.5e18],
    [{ type: "bool", value: true }, false, false],
    [{ type: "tensor", dtype: "torch.bool", value: [true] }, "[false]", [false]],
    [
      { type: "tensor", dtype: "torch.float64", value: [1e18] },
      "[-1.5e18]",
      [-1.5e18],
    ],
  ];
  for (const [info, edited, expected] of cases) {
    const ui = dashboard(info);
    await ui.refresh();
    ui.edit(edited);
    const submission = ui.setAll();
    assert.equal(ui.commands.length, 1);
    assert.deepEqual(ui.commands[0].value, expected);
    ui.complete(0);
    await submission;
    assert.equal(ui.status(), "updated");
  }
});
