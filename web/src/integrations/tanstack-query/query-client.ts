import { QueryClient } from "@tanstack/react-query";

export function createAppQueryClient() {
	return new QueryClient({
		defaultOptions: {
			queries: {
				staleTime: 30_000,
				retry: (failureCount, error) => {
					if (
						error &&
						typeof error === "object" &&
						"status" in error &&
						typeof error.status === "number" &&
						error.status >= 400 &&
						error.status < 500
					) {
						return false;
					}
					return failureCount < 2;
				},
			},
			mutations: {
				retry: false,
			},
		},
	});
}
