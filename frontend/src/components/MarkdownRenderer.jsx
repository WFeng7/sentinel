import React from 'react'

export default function MarkdownRenderer({ content, className = "" }) {
  if (!content) return null

  // Markdown parser for VLM/RAG output
  const parseMarkdown = (text) => {
    if (!text) return ''
    let parsed = String(text)
    // Headers (## Header)
    parsed = parsed.replace(/^###\s+(.+)$/gm, '<h4 class="font-semibold text-sm mt-3 mb-1 text-slate-100">$1</h4>')
    parsed = parsed.replace(/^##\s+(.+)$/gm, '<h3 class="font-semibold text-sm mt-3 mb-1 text-slate-100">$1</h3>')
    parsed = parsed.replace(/^#\s+(.+)$/gm, '<h3 class="font-semibold text-sm mt-3 mb-1 text-slate-100">$1</h3>')
    // Bold (**text**) and italic (*text*)
    parsed = parsed.replace(/\*\*(.*?)\*\*/g, '<strong class="text-slate-100">$1</strong>')
    parsed = parsed.replace(/\*(.*?)\*/g, '<em>$1</em>')
    parsed = parsed.replace(/_(.*?)_/g, '<em>$1</em>')
    // Inline code `code`
    parsed = parsed.replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 rounded bg-slate-800 text-slate-200 text-xs">$1</code>')
    // Bullet lists (- or * at line start)
    parsed = parsed.replace(/^[\-\*]\s+(.+)$/gm, '<li class="ml-4 mb-0.5">$1</li>')
    parsed = parsed.replace(/(<li class="ml-4 mb-0\.5">.*<\/li>\n?)+/gs, (m) => `<ul class="list-disc list-inside mb-2 space-y-0.5">${m}</ul>`)
    // Numbered lists
    parsed = parsed.replace(/^\d+\.\s+(.+)$/gm, '<li class="ml-4 mb-0.5">$1</li>')
    // Paragraphs and line breaks
    parsed = parsed.replace(/\n\n+/g, '</p><p class="mb-2 leading-relaxed">')
    parsed = parsed.replace(/\n/g, '<br />')
    parsed = `<p class="mb-2 leading-relaxed">${parsed}</p>`
    return parsed
  }

  return (
    <div 
      className={`text-sm text-slate-200 prose prose-invert prose-sm max-w-none ${className}`}
      dangerouslySetInnerHTML={{ __html: parseMarkdown(content) }}
    />
  )
}
