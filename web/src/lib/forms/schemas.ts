import { z } from "zod";

export const questionSchema = z.object({
	question: z.string().trim().min(1, "Enter a question."),
});

export const ingestSchema = z.object({
	file: z
		.custom<File>((value) => value instanceof File, "Select a PDF file.")
		.refine(
			(file) => file.name.toLowerCase().endsWith(".pdf"),
			"Only PDF files are supported.",
		),
});

export const textSchema = z.object({
	text: z.string().trim().min(1, "Enter some text."),
});

export const promptTestSchema = z.object({
	template: z.string().min(1, "Choose a template."),
	variables: z
		.array(
			z.object({
				key: z.string(),
				value: z.string(),
			}),
		)
		.min(1),
});
