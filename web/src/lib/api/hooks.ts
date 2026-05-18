import { useMutation, useQuery } from "@tanstack/react-query";
import * as api from "#/lib/api/endpoints.ts";
import type { PromptTestRequest } from "#/lib/api/types.ts";

export const queryKeys = {
	health: ["health"] as const,
};

export function useHealthQuery() {
	return useQuery({
		queryKey: queryKeys.health,
		queryFn: api.health,
		refetchInterval: 30_000,
	});
}

export function useIngestMutation() {
	return useMutation({
		mutationFn: (file: File) => api.ingestPdf(file),
	});
}

export function useAskMutation() {
	return useMutation({
		mutationFn: (input: { question: string; documentId?: string | null }) =>
			api.ask(input.question, input.documentId),
	});
}

export function useEnhanceMutation() {
	return useMutation({
		mutationFn: (question: string) => api.enhance(question),
	});
}

export function useTokenCountMutation() {
	return useMutation({
		mutationFn: (text: string) => api.countTokens(text),
	});
}

export function usePromptTestMutation() {
	return useMutation({
		mutationFn: (request: PromptTestRequest) => api.testPrompt(request),
	});
}

export function useLlmTestMutation() {
	return useMutation({
		mutationFn: (question: string) => api.testLlm(question),
	});
}

export function useSelfConsistencyMutation() {
	return useMutation({
		mutationFn: (question: string) => api.selfConsistency(question),
	});
}
