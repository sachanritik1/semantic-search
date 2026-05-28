import { env } from "#/env.ts";
import type { ApiErrorBody } from "#/lib/api/types.ts";

export class ApiError extends Error {
	status: number;
	body: ApiErrorBody | null;

	constructor(message: string, status: number, body: ApiErrorBody | null = null) {
		super(message);
		this.name = "ApiError";
		this.status = status;
		this.body = body;
	}
}

export function getApiBase(): string {
	return env.VITE_API_URL;
}

function formatDetail(body: ApiErrorBody | null): string {
	if (!body?.detail) {
		return "Request failed";
	}
	if (typeof body.detail === "string") {
		return body.detail;
	}
	return body.detail.map((item) => item.msg).join("; ");
}

export async function apiFetch<T>(
	path: string,
	init?: RequestInit,
): Promise<T> {
	const base = getApiBase();
	const url = `${base}${path.startsWith("/") ? path : `/${path}`}`;
	const response = await fetch(url, init);

	if (!response.ok) {
		let body: ApiErrorBody | null = null;
		try {
			body = (await response.json()) as ApiErrorBody;
		} catch {
			body = null;
		}
		throw new ApiError(formatDetail(body), response.status, body);
	}

	if (response.status === 204) {
		return undefined as T;
	}

	return (await response.json()) as T;
}

export async function apiPostJson<T>(
	path: string,
	body: unknown,
): Promise<T> {
	return apiFetch<T>(path, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
	});
}

export async function apiPostFormData<T>(
	path: string,
	formData: FormData,
): Promise<T> {
	return apiFetch<T>(path, {
		method: "POST",
		body: formData,
	});
}
