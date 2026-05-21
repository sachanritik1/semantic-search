import { apiFetch, apiPostFormData, apiPostJson } from "#/lib/api/client.ts";
import { postSseWithRetry, SseTerminalError } from "#/lib/api/sse.ts";
import type {
	AskResponse,
	AskStreamHandlers,
	AskStreamMeta,
	EnhanceResponse,
	HealthResponse,
	IngestResponse,
	LlmTestResponse,
	PromptTestRequest,
	PromptTestResponse,
	AskRequest,
	QuestionRequest,
	SelfConsistencyResponse,
	TokenCountRequest,
	TokenCountResponse,
} from "#/lib/api/types.ts";

export function health() {
	return apiFetch<HealthResponse>("/health");
}

export function ingestPdf(file: File) {
	const formData = new FormData();
	formData.append("file", file);
	return apiPostFormData<IngestResponse>("/ingest", formData);
}

export function ask(question: string, documentId: string) {
	return apiPostJson<AskResponse>("/ask", {
		question,
		document_id: documentId,
	} satisfies AskRequest);
}

export function askStream(
	question: string,
	documentId: string,
	handlers: AskStreamHandlers,
	signal?: AbortSignal,
) {
	return postSseWithRetry(
		"/ask/stream",
		{
			question,
			document_id: documentId,
		} satisfies AskRequest,
		(event, data) => {
			if (event === "meta") {
				handlers.onMeta?.(data as AskStreamMeta);
				return;
			}
			if (event === "token") {
				const payload = data as { text: string };
				handlers.onToken?.(payload.text);
				return;
			}
			if (event === "done") {
				handlers.onDone?.(data as { cache_hit: boolean });
				return;
			}
			if (event === "error") {
				const payload = data as { message: string };
				handlers.onError?.(payload.message);
			}
		},
		signal,
		{
			onRetry: handlers.onRetry,
		},
	).catch((err) => {
		if (err instanceof SseTerminalError) {
			handlers.onError?.(err.message);
			return;
		}
		throw err;
	});
}

export function enhance(question: string) {
	return apiPostJson<EnhanceResponse>("/enhance", {
		question,
	} satisfies QuestionRequest);
}

export function countTokens(text: string) {
	return apiPostJson<TokenCountResponse>("/tokens/count", {
		text,
	} satisfies TokenCountRequest);
}

export function testPrompt(request: PromptTestRequest) {
	return apiPostJson<PromptTestResponse>("/prompt/test", request);
}

export function testLlm(question: string) {
	return apiPostJson<LlmTestResponse>("/llm/test", {
		question,
	} satisfies QuestionRequest);
}

export function selfConsistency(question: string) {
	return apiPostJson<SelfConsistencyResponse>("/self-consistency", {
		question,
	} satisfies QuestionRequest);
}
