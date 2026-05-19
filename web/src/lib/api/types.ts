export interface QuestionRequest {
	question: string;
}

export interface AskRequest {
	question: string;
	document_id: string;
}

export interface HealthResponse {
	status: string;
}

export interface AskResponse {
	response: string;
	original_question: string;
	enhanced_question: string;
	enhanced_questions?: string[];
	cache_hit?: boolean;
}

export interface EnhanceResponse {
	original: string;
	enhanced: string;
	enhanced_queries: string[];
}

export interface IngestResponse {
	document_id: string;
	chunks_total: number;
	chunks_saved: number;
}

export interface TokenCountRequest {
	text: string;
}

export interface TokenCountResponse {
	token_count: number;
	tokens: number[];
}

export interface PromptTestRequest {
	template: string;
	variables: Record<string, string>;
}

export interface PromptTestResponse {
	response?: string;
	error?: string;
}

export interface LlmResponsePayload {
	content: string;
	model: string | null;
	usage: Record<string, unknown> | null;
	raw_response: unknown;
}

export interface LlmTestResponse {
	response: LlmResponsePayload;
}

export interface SelfConsistencyResponse {
	final_answer: string;
}

export interface ApiErrorBody {
	detail?: string | { msg: string; type: string }[];
}
