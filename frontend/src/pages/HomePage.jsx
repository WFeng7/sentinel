export default function HomePage() {
  return (
    <section className="mt-10 space-y-6">
      <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-8 text-slate-200">
        <h2 className="text-lg font-semibold text-white">Placeholder content</h2>
        <p className="mt-2 text-sm text-slate-300">
          This page is intentionally taller now so it scrolls on mobile. Replace these sections with real content
          when ready.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-8 text-slate-200">
          <div className="text-xs uppercase tracking-[0.2em] text-slate-500">Section</div>
          <h3 className="mt-3 text-base font-semibold text-white">Road monitoring</h3>
          <div className="mt-4 h-40 rounded-xl border border-slate-800 bg-slate-950" />
        </div>
        <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-8 text-slate-200">
          <div className="text-xs uppercase tracking-[0.2em] text-slate-500">Section</div>
          <h3 className="mt-3 text-base font-semibold text-white">Alerts</h3>
          <div className="mt-4 h-40 rounded-xl border border-slate-800 bg-slate-950" />
        </div>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-8 text-slate-200">
        <h3 className="text-base font-semibold text-white">Extra padding</h3>
        <p className="mt-2 text-sm text-slate-300">
          Spacer content below to ensure consistent scrolling behavior across devices.
        </p>
        <div className="mt-6 h-[60vh] rounded-xl border border-slate-800 bg-slate-950" />
      </div>
    </section>
  )
}
