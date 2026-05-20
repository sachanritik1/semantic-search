import { ApiError, getApiBase } from "#/lib/api/client.ts";
import type { ApiErrorBody } from "#/lib/api/types.ts";

export type SseEventHandler = (event: string, data: unknown) => void;

async function formatApiError(response: Response): Promise<ApiError> {
	let body: ApiErrorBody | null = null;
	try {
		body = (await response.json()) as ApiErrorBody;
	} catch {
		body = null;
	}
	const detail =
		typeof body?.detail === "string"
			? body.detail
			: body?.detail?.map((item) => item.msg).join("; ") ?? "Request failed";
	return new ApiError(detail, response.status, body);
}

function dispatchSseBlock(block: string, onEvent: SseEventHandler): void {
	let eventName = "message";
	const dataLines: string[] = [];

	for (const line of block.split("\n")) {
		if (line.startsWith("event:")) {
			eventName = line.slice(6).trim();
		} else if (line.startsWith("data:")) {
			dataLines.push(line.slice(5).trimStart());
		}
	}

	if (dataLines.length === 0) {
		return;
	}

	const raw = dataLines.join("\n");
	onEvent(eventName, JSON.parse(raw) as unknown);
}

export async function consumeSseStream(
	response: Response,
	onEvent: SseEventHandler,
): Promise<void> {
	if (!response.ok) {
		throw await formatApiError(response);
	}

	const reader = response.body?.getReader();
	if (!reader) {
		throw new Error("Response body is not readable");
	}

	const decoder = new TextDecoder();
	let buffer = "";

	while (true) {
		const { done, value } = await reader.read();
		if (done) {
			break;
		}

		buffer += decoder.decode(value, { stream: true });
		const parts = buffer.split("\n\n");
		buffer = parts.pop() ?? "";

		for (const part of parts) {
			const block = part.trim();
			if (block) {
				dispatchSseBlock(block, onEvent);
			}
		}
	}

	const tail = buffer.trim();
	if (tail) {
		dispatchSseBlock(tail, onEvent);
	}
}

export async function postSse(
	path: string,
	body: unknown,
	onEvent: SseEventHandler,
	signal?: AbortSignal,
): Promise<void> {
	const base = getApiBase();
	const url = `${base}${path.startsWith("/") ? path : `/${path}`}`;
	const response = await fetch(url, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
		signal,
	});
	await consumeSseStream(response, onEvent);
}
