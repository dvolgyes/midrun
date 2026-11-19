"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const { setImmediate } = require("node:timers/promises");
const { dashboard } = require("./dashboard_fixture.cjs");

test("refresh cannot make a successful edit replayable", async () => {
  const ui = dashboard();
  await ui.refresh();
  ui.edit("17");
  const pending = ui.setAll();
  await ui.refresh();
  assert.equal(ui.value(), "17");
  assert.equal(ui.status(), "setting…");

  ui.complete(0);
  await pending;
  ui.setFromServer(39);
  await ui.refresh();
  assert.equal(ui.value(), "39");
  ui.setAll();
  assert.equal(ui.status(), "updated");
  assert.equal(ui.commands.length, 1);
});

test("a long-queued edit cannot be submitted again", async () => {
  const ui = dashboard();
  await ui.refresh();
  ui.edit("17");
  const pending = ui.setAll();
  await setImmediate();
  ui.holdFor(0, 61_000);
  await setImmediate();

  await ui.refresh();
  assert.equal(ui.value(), "17");
  ui.setAll();
  assert.equal(ui.commands.length, 1);
  assert.equal(ui.status(), "setting…");

  ui.complete(0);
  await pending;
  ui.setFromServer(39);
  await ui.refresh();
  assert.equal(ui.value(), "39");
  ui.setAll();
  assert.equal(ui.commands.length, 1);
});

test("a status error cannot make an accepted edit replayable", async () => {
  const ui = dashboard();
  await ui.refresh();
  ui.edit("17");
  const pending = ui.setAll();
  ui.failPoll(0);
  await pending;
  await ui.refresh();
  assert.equal(ui.status(), "error checking command 0: connection closed");
  ui.setAll();
  assert.equal(ui.commands.length, 1);

  ui.edit("23");
  const next = ui.setAll();
  assert.equal(ui.commands.length, 2);
  assert.equal(String(ui.commands[1].value), "23");
  ui.complete(1);
  await next;
});

test("a pending edit is sent once and cannot clear a newer edit", async () => {
  const ui = dashboard();
  await ui.refresh();
  ui.edit("17");
  const pending = ui.setAll();
  await ui.refresh();
  ui.setAll();
  assert.equal(ui.commands.length, 1);
  ui.edit("23");
  ui.setAll();
  assert.equal(ui.commands.length, 1);

  ui.complete(0);
  await pending;
  await ui.refresh();
  assert.equal(ui.value(), "23");
  assert.equal(ui.status(), "");
  const next = ui.setAll();
  assert.deepEqual(ui.commands.map(({ value }) => String(value)), ["17", "23"]);
  ui.complete(1);
  await next;
  assert.equal(ui.status(), "updated");
});

test("a failed submission leaves the edit available for retry", async () => {
  const ui = dashboard();
  await ui.refresh();
  ui.edit("17");
  const pending = ui.setAll();
  await ui.refresh();
  ui.complete(0, "failed");
  await pending;
  assert.equal(ui.status(), "failed: rejected");
  await ui.refresh();
  assert.equal(ui.value(), "17");
  const retry = ui.setAll();
  assert.equal(ui.commands.length, 2);
  assert.equal(String(ui.commands[1].value), "17");
  ui.complete(1);
  await retry;
});
