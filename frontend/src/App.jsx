export default function App() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <div className="mx-auto max-w-3xl px-6 py-16">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-8 shadow-2xl shadow-slate-900/40">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Sentinel</p>
          <h1 className="mt-4 text-3xl font-semibold text-white">React + Tailwind is live</h1>
          <p className="mt-3 text-slate-300">
            Frontend runs on Vite. Backend runs on FastAPI. Use the root dev script to launch both.
          </p>
          <div className="mt-6 flex flex-wrap gap-3 text-sm">
            <span className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-slate-200">Vite</span>
            <span className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-slate-200">React</span>
            <span className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-slate-200">Tailwind</span>
          </div>
        </div>
      </div>
    </main>
  )
}
