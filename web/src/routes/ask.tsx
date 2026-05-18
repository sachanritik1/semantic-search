import { createFileRoute, redirect } from "@tanstack/react-router";

export const Route = createFileRoute("/ask")({
	beforeLoad: () => {
		throw redirect({ to: "/" });
	},
});
