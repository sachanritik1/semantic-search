import { Alert, AlertDescription, AlertTitle } from "#/components/ui/alert.tsx";
import { ApiError } from "#/lib/api/client.ts";

export function ApiErrorAlert({ error }: { error: unknown }) {
	const message =
		error instanceof ApiError
			? error.message
			: error instanceof Error
				? error.message
				: "Something went wrong";

	return (
		<Alert className="border-destructive/40 bg-destructive/5 text-destructive">
			<AlertTitle>Request failed</AlertTitle>
			<AlertDescription>{message}</AlertDescription>
		</Alert>
	);
}
