import { createFileRoute, redirect } from "@tanstack/react-router";

export const Route = createFileRoute("/ingest")({
	beforeLoad: () => {
		throw redirect({ to: "/" });
	},
});
