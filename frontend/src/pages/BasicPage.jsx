export default function BasicPage({ title, children }) {
  return (
    <section className="mt-10">
      <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-8">
        <h1 className="text-2xl font-semibold text-white">{title}</h1>
        <div className="mt-4 text-sm text-slate-300">{children}</div>
      </div>
    </section>
  )
}
