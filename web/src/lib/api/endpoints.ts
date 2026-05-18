import { apiFetch, apiPostFormData, apiPostJson } from "#/lib/api/client.ts";
import type {
	AskResponse,
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
