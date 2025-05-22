// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

"use client";

import Mention from "@tiptap/extension-mention";
import { type Content } from "@tiptap/react";
import {
  EditorContent,
  type EditorInstance,
  EditorRoot,
  type JSONContent,
  StarterKit,
} from "novel";
import { Markdown } from "tiptap-markdown";
import { useDebouncedCallback } from "use-debounce";

import "~/styles/prosemirror.css";
import { resourceSuggestion } from "./resource-suggestion";

const extensions = [
  StarterKit,
  Mention.configure({
    HTMLAttributes: {
      class: "mention",
    },
    suggestion: resourceSuggestion,
  }),
  Markdown.configure({
    html: true,
    tightLists: true,
    tightListClass: "tight",
    bulletListMarker: "-",
    linkify: false,
    breaks: false,
    transformPastedText: false,
    transformCopiedText: false,
  }),
];

export interface MessageInputProps {
  content: Content;
  onChange?: (markdown: string) => void;
}

const MessageInput = ({ content, onChange }: MessageInputProps) => {
  const debouncedUpdates = useDebouncedCallback(
    async (editor: EditorInstance) => {
      if (onChange) {
        const markdown = editor.storage.markdown.getMarkdown();
        onChange(markdown);
      }
    },
    200,
  );

  return (
    <div className="relative w-full">
      <EditorRoot>
        <EditorContent
          immediatelyRender={false}
          extensions={extensions}
          initialContent={(content ?? "") as JSONContent}
          className="border-muted relative h-full w-full"
          editorProps={{
            attributes: {
              class:
                "prose prose-base prose-p:my-4 dark:prose-invert prose-headings:font-title font-default focus:outline-none max-w-full",
            },
          }}
          onUpdate={({ editor }) => {
            debouncedUpdates(editor);
          }}
        ></EditorContent>
      </EditorRoot>
    </div>
  );
};

export default MessageInput;
