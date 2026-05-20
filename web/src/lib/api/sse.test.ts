import { describe, expect, it, vi } from "vitest";
import { consumeSseStream } from "#/lib/api/sse.ts";

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

	it("throws ApiError on non-OK responses", async () => {
		const fetchMock = vi.fn();
		vi.stubGlobal(
			"fetch",
			fetchMock.mockResolvedValue({
				ok: false,
				status: 422,
				json: async () => ({ detail: "validation error" }),
			}),
		);

		const response = new Response(null, {
			status: 422,
			headers: { "Content-Type": "application/json" },
		});

		await expect(
			consumeSseStream(response, () => {}),
		).rejects.toMatchObject({ status: 422 });

		vi.unstubAllGlobals();
	});
});
