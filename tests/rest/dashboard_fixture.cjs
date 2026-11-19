"use strict";

const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const app = readFileSync(
  path.join(__dirname, "../../midrun/rest/static/app.js"),
  "utf8",
);

class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.dataset = {};
    this.textContent = "";
    this.placeholder = "";
    this.listeners = new Map();
  }

  append(child) {
    this.children.push(child);
  }

  replaceChildren() {
    this.children.length = 0;
  }

  querySelectorAll(selector) {
    const matches = {
      "[data-hparam-name]": (node) => "hparamName" in node.dataset,
      "input, textarea": (node) => ["input", "textarea"].includes(node.tag),
      ".status-line": (node) => node.className === "status-line",
    }[selector];
    return this.children.flatMap((node) => [
      ...(matches(node) ? [node] : []),
      ...node.querySelectorAll(selector),
    ]);
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0];
  }

  addEventListener(event, listener) {
    this.listeners.set(event, listener);
  }

  emit(event) {
    return this.listeners.get(event)?.({ target: this });
  }
}

function pollResponse() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function dashboard(info = { type: "int", value: 10 }, name = "hparams.count") {
  let current = info.value;
  let time = 0;
  const nodes = new Map();
  const byId = (id) => {
    if (!nodes.has(id)) nodes.set(id, new Element("div"));
    return nodes.get(id);
  };
  const commands = [];
  const response = (body) => ({
    ok: true,
    headers: { get: () => "application/json" },
    json: async () => body,
    text: async () => JSON.stringify(body),
  });
  const context = vm.createContext({
    Date: { now: () => time },
    setTimeout: queueMicrotask,
    document: {
      getElementById: byId,
      createElement: (tag) => new Element(tag),
      addEventListener() {},
    },
    fetch: async (url, opts) => {
      if (url === "/commands") {
        assert.equal(opts.method, "POST");
        const command = JSON.parse(opts.body);
        assert.equal(command.kind, "set_hparam");
        assert.equal(command.args.name, name);
        commands.push({
          value: command.args.value,
          body: opts.body,
          poll: pollResponse(),
        });
        return response({ id: commands.length - 1, status: "queued" });
      }
      if (url.startsWith("/commands/")) {
        const command = commands[Number(url.split("/").at(-1))];
        const result = await command.poll.promise;
        if (result instanceof Error) throw result;
        return response(result);
      }
      if (url === "/files/model/tunable-hparams.json") {
        return response({ [name]: { ...info, value: current } });
      }
      if (url === "/files/state/dataloader/finite-check.json") {
        return response({ enabled: false });
      }
      if (url === "/files/model/hparams.json") {
        return response({ [name.replace(/^hparams\./, "")]: current });
      }
      throw new Error(`Unexpected request: ${url}`);
    },
  });
  vm.runInContext(app, context);
  context.setupControls();
  const list = byId("tunable-hparams-list");
  const input = () => list.querySelector("input, textarea");
  return {
    commands,
    refresh: () => context.refreshControlsState(),
    value: () => String(input().value),
    placeholder: () => input().placeholder,
    checked: () => input().checked,
    status: () => list.querySelector(".status-line").textContent,
    edit(value) {
      const node = input();
      if (info.type === "bool") node.checked = Boolean(value);
      else node.value = String(value);
      node.emit("input");
    },
    setAll: () => byId("tunable-hparams-set-all-btn").emit("click"),
    setFromServer(value) {
      current = value;
    },
    holdFor(index, elapsedMs) {
      time += elapsedMs;
      const command = commands[index];
      const previous = command.poll;
      command.poll = pollResponse();
      previous.resolve({ status: "queued" });
    },
    failPoll(index) {
      commands[index].poll.resolve(new Error("connection closed"));
    },
    complete(index, status = "succeeded") {
      const command = commands[index];
      if (status === "succeeded") current = command.value;
      command.poll.resolve({ status, error: "rejected" });
    },
  };
}

module.exports = { dashboard };
