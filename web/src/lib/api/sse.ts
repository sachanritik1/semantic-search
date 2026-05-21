import { ApiError, getApiBase } from "#/lib/api/client.ts";
import type { ApiErrorBody } from "#/lib/api/types.ts";

export type SseEventHandler = (event: string, data: unknown) => void;

export class SseIdleTimeoutError extends Error {
	constructor(message = "Stream idle timeout") {
		super(message);
		this.name = "SseIdleTimeoutError";
	}
}

export class SseTerminalError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "SseTerminalError";
	}
}

const DEFAULT_IDLE_TIMEOUT_MS = 30_000;
const DEFAULT_MAX_RETRIES = 2;
const DEFAULT_BASE_DELAY_MS = 500;
const DEFAULT_BACKOFF_FACTOR = 2;

export interface ConsumeSseStreamOptions {
	idleTimeoutMs?: number;
	signal?: AbortSignal;
}

export interface PostSseWithRetryOptions {
	maxRetries?: number;
	baseDelayMs?: number;
	backoffFactor?: number;
	idleTimeoutMs?: number;
	onRetry?: (attempt: number, reason: string) => void;
}

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

function isAbortError(err: unknown): boolean {
	return err instanceof DOMException && err.name === "AbortError";
}

function backoffDelayMs(
	attempt: number,
	baseDelayMs: number,
	backoffFactor: number,
): number {
	const exponential = baseDelayMs * backoffFactor ** attempt;
	const jitter = exponential * (0.5 + Math.random());
	return Math.round(jitter);
}

function isRetryableError(err: unknown): boolean {
	if (err instanceof SseIdleTimeoutError) {
		return true;
	}
	if (err instanceof TypeError) {
		return true;
	}
	if (err instanceof ApiError) {
		return err.status >= 500 || err.status === 429;
	}
	return false;
}

function retryReason(err: unknown): string {
	if (err instanceof SseIdleTimeoutError) {
		return "idle timeout";
	}
	if (err instanceof ApiError) {
		return `HTTP ${err.status}`;
	}
	if (err instanceof TypeError) {
		return "network error";
	}
	return "connection error";
}

async function readWithIdleTimeout(
	reader: ReadableStreamDefaultReader<Uint8Array>,
	idleTimeoutMs: number,
	abortController: AbortController,
): Promise<ReadableStreamReadResult<Uint8Array>> {
	let timeoutId: ReturnType<typeof setTimeout> | undefined;

	const timeoutPromise = new Promise<never>((_, reject) => {
		timeoutId = setTimeout(() => {
			abortController.abort();
			reject(new SseIdleTimeoutError());
		}, idleTimeoutMs);
	});

	try {
		return await Promise.race([reader.read(), timeoutPromise]);
	} finally {
		if (timeoutId !== undefined) {
			clearTimeout(timeoutId);
		}
	}
}

export async function consumeSseStream(
	response: Response,
	onEvent: SseEventHandler,
	options: ConsumeSseStreamOptions = {},
): Promise<void> {
	if (!response.ok) {
		throw await formatApiError(response);
	}

	const reader = response.body?.getReader();
	if (!reader) {
		throw new Error("Response body is not readable");
	}

	const idleTimeoutMs = options.idleTimeoutMs ?? DEFAULT_IDLE_TIMEOUT_MS;
	const abortController = new AbortController();
	const externalSignal = options.signal;

	if (externalSignal?.aborted) {
		throw new DOMException("Aborted", "AbortError");
	}

	const onExternalAbort = () => abortController.abort();
	externalSignal?.addEventListener("abort", onExternalAbort);

	const decoder = new TextDecoder();
	let buffer = "";
	let terminalError: SseTerminalError | null = null;

	const wrappedOnEvent: SseEventHandler = (event, data) => {
		if (event === "error") {
			const payload = data as { message?: string };
			terminalError = new SseTerminalError(
				payload.message ?? "Stream failed",
			);
			return;
		}
		onEvent(event, data);
	};

	try {
		while (true) {
			if (abortController.signal.aborted && !terminalError) {
				if (externalSignal?.aborted) {
					throw new DOMException("Aborted", "AbortError");
				}
			}

			const { done, value } = await readWithIdleTimeout(
				reader,
				idleTimeoutMs,
				abortController,
			);
			if (done) {
				break;
			}

			buffer += decoder.decode(value, { stream: true });
			const parts = buffer.split("\n\n");
			buffer = parts.pop() ?? "";

			for (const part of parts) {
				const block = part.trim();
				if (block) {
					dispatchSseBlock(block, wrappedOnEvent);
					if (terminalError) {
						throw terminalError;
					}
				}
			}
		}

		const tail = buffer.trim();
		if (tail) {
			dispatchSseBlock(tail, wrappedOnEvent);
		}

		if (terminalError) {
			throw terminalError;
		}
	} finally {
		externalSignal?.removeEventListener("abort", onExternalAbort);
		try {
			reader.releaseLock();
		} catch {
			// already released
		}
	}
}

export async function postSse(
	path: string,
	body: unknown,
	onEvent: SseEventHandler,
	signal?: AbortSignal,
	options: Pick<ConsumeSseStreamOptions, "idleTimeoutMs"> = {},
): Promise<void> {
	const base = getApiBase();
	const url = `${base}${path.startsWith("/") ? path : `/${path}`}`;
	const response = await fetch(url, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
		signal,
	});
	await consumeSseStream(response, onEvent, {
		idleTimeoutMs: options.idleTimeoutMs,
		signal,
	});
}

function delay(ms: number, signal?: AbortSignal): Promise<void> {
	return new Promise((resolve, reject) => {
		if (signal?.aborted) {
			reject(new DOMException("Aborted", "AbortError"));
			return;
		}
		const timeoutId = setTimeout(resolve, ms);
		const onAbort = () => {
			clearTimeout(timeoutId);
			reject(new DOMException("Aborted", "AbortError"));
		};
		signal?.addEventListener("abort", onAbort, { once: true });
	});
}

export async function postSseWithRetry(
	path: string,
	body: unknown,
	onEvent: SseEventHandler,
	externalSignal?: AbortSignal,
	options: PostSseWithRetryOptions = {},
): Promise<void> {
	const maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
	const baseDelayMs = options.baseDelayMs ?? DEFAULT_BASE_DELAY_MS;
	const backoffFactor = options.backoffFactor ?? DEFAULT_BACKOFF_FACTOR;
	const idleTimeoutMs = options.idleTimeoutMs ?? DEFAULT_IDLE_TIMEOUT_MS;

	let lastError: unknown;

	for (let attempt = 0; attempt <= maxRetries; attempt++) {
		if (externalSignal?.aborted) {
			throw new DOMException("Aborted", "AbortError");
		}

		const attemptController = new AbortController();
		const onExternalAbort = () => attemptController.abort();
		externalSignal?.addEventListener("abort", onExternalAbort);

		try {
			await postSse(path, body, onEvent, attemptController.signal, {
				idleTimeoutMs,
			});
			return;
		} catch (err) {
			lastError = err;

			if (isAbortError(err) && externalSignal?.aborted) {
				throw err;
			}
			if (err instanceof SseTerminalError) {
				throw err;
			}
			if (!isRetryableError(err) || attempt >= maxRetries) {
				throw err;
			}

			const reason = retryReason(err);
			options.onRetry?.(attempt + 1, reason);

			const waitMs = backoffDelayMs(attempt, baseDelayMs, backoffFactor);
			await delay(waitMs, externalSignal);
		} finally {
			externalSignal?.removeEventListener("abort", onExternalAbort);
		}
	}

	throw lastError;
}
