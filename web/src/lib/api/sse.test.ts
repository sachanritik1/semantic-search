import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "#/lib/api/client.ts";
import {
	consumeSseStream,
	postSseWithRetry,
	SseIdleTimeoutError,
	SseTerminalError,
} from "#/lib/api/sse.ts";

function sseBody(events: { event: string; data: unknown }[]): string {
	return events
		.map(
			({ event, data }) =>
				`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`,
		)
		.join("");
}

describe("consumeSseStream", () => {
	it("parses named SSE events from a streamed response", async () => {
		const received: { event: string; data: unknown }[] = [];
		const body = sseBody([
			{ event: "meta", data: { original_question: "q" } },
			{ event: "token", data: { text: "Hi" } },
			{ event: "done", data: { cache_hit: false } },
		]);

		const stream = new ReadableStream({
			start(controller) {
				controller.enqueue(new TextEncoder().encode(body));
				controller.close();
			},
		});

		const response = new Response(stream, {
			status: 200,
			headers: { "Content-Type": "text/event-stream" },
		});

		await consumeSseStream(response, (event, data) => {
			received.push({ event, data });
		});

		expect(received).toEqual([
			{ event: "meta", data: { original_question: "q" } },
			{ event: "token", data: { text: "Hi" } },
			{ event: "done", data: { cache_hit: false } },
		]);
	});

	it("handles events split across chunks", async () => {
		const received: string[] = [];
		const payload = sseBody([{ event: "token", data: { text: "ab" } }]);
		const part1 = payload.slice(0, 12);
		const part2 = payload.slice(12);

		const stream = new ReadableStream({
			start(controller) {
				controller.enqueue(new TextEncoder().encode(part1));
				controller.enqueue(new TextEncoder().encode(part2));
				controller.close();
			},
		});

		const response = new Response(stream, { status: 200 });
		await consumeSseStream(response, (event, data) => {
			if (event === "token") {
				received.push((data as { text: string }).text);
			}
		});

		expect(received).toEqual(["ab"]);
	});

	it("ignores heartbeat comment frames", async () => {
		const received: string[] = [];
		const body = `: ping\n\n${sseBody([{ event: "token", data: { text: "x" } }])}`;

		const stream = new ReadableStream({
			start(controller) {
				controller.enqueue(new TextEncoder().encode(body));
				controller.close();
			},
		});

		await consumeSseStream(new Response(stream, { status: 200 }), (event, data) => {
			if (event === "token") {
				received.push((data as { text: string }).text);
			}
		});

		expect(received).toEqual(["x"]);
	});

	it("throws ApiError on non-OK responses", async () => {
		const response = new Response(null, {
			status: 422,
			headers: { "Content-Type": "application/json" },
		});
		Object.defineProperty(response, "ok", { value: false });
		Object.defineProperty(response, "json", {
			value: async () => ({ detail: "validation error" }),
		});

		await expect(
			consumeSseStream(response, () => {}),
		).rejects.toMatchObject({ status: 422 });
	});

	it("throws SseIdleTimeoutError when stream stalls", async () => {
		const stream = new ReadableStream({
			start() {
				// never enqueue or close
			},
		});

		await expect(
			consumeSseStream(new Response(stream, { status: 200 }), () => {}, {
				idleTimeoutMs: 50,
			}),
		).rejects.toBeInstanceOf(SseIdleTimeoutError);
	});

	it("throws SseTerminalError on event error frame", async () => {
		const body = sseBody([{ event: "error", data: { message: "LLM failed" } }]);
		const stream = new ReadableStream({
			start(controller) {
				controller.enqueue(new TextEncoder().encode(body));
				controller.close();
			},
		});

		await expect(
			consumeSseStream(new Response(stream, { status: 200 }), () => {}),
		).rejects.toMatchObject({ message: "LLM failed" });
	});
});

describe("postSseWithRetry", () => {
	afterEach(() => {
		vi.unstubAllGlobals();
		vi.useRealTimers();
	});

	it("retries on network failure then succeeds", async () => {
		vi.useFakeTimers();
		const okBody = sseBody([{ event: "done", data: { cache_hit: false } }]);
		let calls = 0;

		vi.stubGlobal("fetch", vi.fn(async () => {
			calls += 1;
			if (calls < 3) {
				throw new TypeError("Failed to fetch");
			}
			return new Response(okBody, {
				status: 200,
				headers: { "Content-Type": "text/event-stream" },
			});
		}));

		const onRetry = vi.fn();
		const done = postSseWithRetry(
			"/ask/stream",
			{ question: "q", document_id: "d" },
			() => {},
			undefined,
			{
				maxRetries: 2,
				baseDelayMs: 100,
				backoffFactor: 2,
				onRetry,
			},
		);

		await vi.runAllTimersAsync();
		await done;

		expect(calls).toBe(3);
		expect(onRetry).toHaveBeenCalledTimes(2);
	});

	it("does not retry on 4xx responses", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async () =>
				new Response(JSON.stringify({ detail: "bad request" }), {
					status: 400,
					headers: { "Content-Type": "application/json" },
				}),
			),
		);

		await expect(
			postSseWithRetry(
				"/ask/stream",
				{ question: "q", document_id: "d" },
				() => {},
			),
		).rejects.toBeInstanceOf(ApiError);

		expect(fetch).toHaveBeenCalledTimes(1);
	});

	it("aborts during backoff without further fetch attempts", async () => {
		vi.useFakeTimers();
		const controller = new AbortController();
		let calls = 0;

		vi.stubGlobal("fetch", vi.fn(async () => {
			calls += 1;
			throw new TypeError("Failed to fetch");
		}));

		const promise = postSseWithRetry(
			"/ask/stream",
			{ question: "q", document_id: "d" },
			() => {},
			controller.signal,
			{ maxRetries: 2, baseDelayMs: 500 },
		);

		await vi.advanceTimersByTimeAsync(0);
		controller.abort();

		await expect(promise).rejects.toMatchObject({ name: "AbortError" });
		expect(calls).toBe(1);
	});
});

describe("SseTerminalError", () => {
	it("has the expected name", () => {
		expect(new SseTerminalError("x").name).toBe("SseTerminalError");
	});
});
