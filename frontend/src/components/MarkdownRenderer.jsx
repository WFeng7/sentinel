import React from 'react'

export default function MarkdownRenderer({ content, className = "" }) {
  if (!content) return null

  // Simple markdown parser for our use case
  const parseMarkdown = (text) => {
    // Handle bold text (**text**)
    let parsed = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    
    // Handle headers (## Header)
    parsed = parsed.replace(/^##\s+(.+)$/gm, '<h3 class="font-semibold text-sm mt-3 mb-1 text-slate-100">$1</h3>')
    
    // Handle line breaks
    parsed = parsed.replace(/\n\n/g, '</p><p class="mb-2">')
    parsed = parsed.replace(/\n/g, '<br />')
    
    // Wrap in paragraphs
    parsed = `<p class="mb-2">${parsed}</p>`
    
    return parsed
  }

  return (
    <div 
      className={`text-sm text-slate-200 prose prose-invert prose-sm max-w-none ${className}`}
      dangerouslySetInnerHTML={{ __html: parseMarkdown(content) }}
    />
  )
}
