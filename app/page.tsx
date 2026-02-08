'use client'

import { useState } from 'react'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'YOUR_MODAL_URL_HERE'

export default function Home() {
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<any>(null)
  const [error, setError] = useState('')

  const examples = ['headache', 'stomach pain', 'fever', 'loss of appetite', 'cough']

  const analyze = async () => {
    if (!input.trim()) return

    setLoading(true)
    setError('')
    setResults(null)

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 120000)

    try {
      const response = await fetch(API_BASE, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: input }),
        signal: controller.signal
      })

      clearTimeout(timeout)

      if (!response.ok) throw new Error(`HTTP ${response.status}`)

      const data = await response.json()
      setResults(data)
    } catch (err: any) {
      setError(err.name === 'AbortError' ? 'Request timeout. AI is processing, please try again.' : 'Cannot connect to backend. Check Modal deployment.')
    } finally {
      setLoading(false)
    }
  }

  const treatment = results?.results?.[0]?.treatment_info

  return (
    <main style={{ minHeight: '100vh', background: 'linear-gradient(to bottom, #f0f7e8, white)', padding: '2rem 1rem' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#4a7c28', letterSpacing: '0.1em', marginBottom: '0.5rem' }}>
            🌿 WHO ITA · SNOMED CT
          </div>
          <h1 style={{ fontSize: '3rem', fontWeight: 700, color: '#2d5016', marginBottom: '0.5rem' }}>
            AyurAssist
          </h1>
          <p style={{ color: '#666' }}>AI-Powered Ayurveda Clinical Decision Support</p>
        </div>

        {/* Search */}
        <div style={{ background: 'white', borderRadius: '1.5rem', padding: '2rem', boxShadow: '0 4px 16px rgba(0,0,0,0.1)', marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Describe your symptoms</h2>
          <p style={{ color: '#888', fontSize: '0.875rem', marginBottom: '1rem' }}>
            Enter symptoms — AI will analyze with Bio_ClinicalBERT & AyurParam
          </p>
          
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
            {examples.map(ex => (
              <button key={ex} onClick={() => setInput(ex)}
                style={{ padding: '0.5rem 1rem', background: '#f5f5f5', border: 'none', borderRadius: '2rem', cursor: 'pointer', fontSize: '0.875rem' }}>
                {ex}
              </button>
            ))}
          </div>

          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && analyze()}
              placeholder="e.g., I have severe headache and nausea"
              style={{ flex: 1, padding: '0.875rem 1.25rem', fontSize: '1rem', border: '2px solid #ddd', borderRadius: '0.75rem', outline: 'none' }}
            />
            <button
              onClick={analyze}
              disabled={loading || !input.trim()}
              style={{
                padding: '0.875rem 2rem', background: loading ? '#ccc' : 'linear-gradient(135deg, #4a7c28, #5a9632)',
                color: 'white', border: 'none', borderRadius: '0.75rem', fontSize: '1rem', fontWeight: 600, cursor: loading ? 'default' : 'pointer'
              }}
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
          </div>
        </div>

        {/* Loading */}
        {loading && (
          <div style={{ background: '#e3f2fd', border: '1px solid #90caf9', borderRadius: '1rem', padding: '1.5rem', textAlign: 'center' }}>
            <div style={{ display: 'inline-block', width: '2rem', height: '2rem', border: '3px solid #1976d2', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite', marginBottom: '0.5rem' }}></div>
            <p style={{ color: '#1976d2' }}>AI is analyzing... This may take 30-90s</p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div style={{ background: '#ffebee', border: '1px solid #ef5350', borderRadius: '1rem', padding: '1rem', color: '#c62828' }}>
            ⚠️ {error}
          </div>
        )}

        {/* Results */}
        {results && treatment && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            
            {/* Clinical Info */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div style={{ background: '#e3f2fd', borderRadius: '1rem', padding: '1.5rem' }}>
                <h3 style={{ fontSize: '0.875rem', fontWeight: 700, color: '#1976d2', marginBottom: '1rem' }}>CLINICAL ENTITIES</h3>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {results.clinical_entities?.map((ent: any, i: number) => (
                    <span key={i} style={{ padding: '0.5rem 1rem', background: 'white', border: '1px solid #90caf9', borderRadius: '2rem', fontSize: '0.875rem', color: '#1976d2' }}>
                      {ent.word}
                    </span>
                  ))}
                </div>
              </div>

              <div style={{ background: '#f3e5f5', borderRadius: '1rem', padding: '1.5rem' }}>
                <h3 style={{ fontSize: '0.875rem', fontWeight: 700, color: '#7b1fa2', marginBottom: '1rem' }}>MEDICAL CODES</h3>
                <div style={{ fontSize: '0.875rem', color: '#555' }}>
                  <div style={{ marginBottom: '0.5rem' }}><strong>UMLS:</strong> {results.umls_cui || 'N/A'}</div>
                  <div><strong>SNOMED:</strong> {results.snomed_code || results.results?.[0]?.snomed_code || 'N/A'}</div>
                </div>
              </div>
            </div>

            {/* Main Condition */}
            <div style={{ background: 'linear-gradient(135deg, #4a7c28, #5a9632)', color: 'white', borderRadius: '1rem', padding: '2rem' }}>
              <h2 style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '0.5rem' }}>{treatment.condition_name}</h2>
              <p style={{ fontSize: '1.25rem', fontStyle: 'italic', opacity: 0.9 }}>{treatment.sanskrit_name}</p>
            </div>

            {/* Description */}
            <Card title="📋 Clinical Overview">
              <p style={{ fontSize: '1rem', lineHeight: 1.7, color: '#444' }}>{treatment.brief_description}</p>
              <div style={{ marginTop: '1rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div style={{ padding: '0.75rem', background: '#f5f5f5', borderRadius: '0.5rem' }}>
                  <strong>Dosha:</strong> {treatment.dosha_involvement}
                </div>
                <div style={{ padding: '0.75rem', background: '#f5f5f5', borderRadius: '0.5rem' }}>
                  <strong>Prognosis:</strong> {treatment.prognosis}
                </div>
              </div>
            </Card>

            {/* Causes & Symptoms */}
            {treatment.nidana_causes?.length > 0 && (
              <Card title="🔍 Nidana (Causes)">
                <ul style={{ paddingLeft: '1.5rem', lineHeight: 1.8 }}>
                  {treatment.nidana_causes.map((c: string, i: number) => <li key={i}>{c}</li>)}
                </ul>
              </Card>
            )}

            {treatment.rupa_symptoms?.length > 0 && (
              <Card title="🩺 Rupa (Symptoms)">
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {treatment.rupa_symptoms.map((s: string, i: number) => (
                    <span key={i} style={{ padding: '0.5rem 1rem', background: '#f0f7e8', border: '1px solid #c5e1a5', borderRadius: '2rem', fontSize: '0.875rem' }}>
                      {s}
                    </span>
                  ))}
                </div>
              </Card>
            )}

            {/* Remedies */}
            {treatment.ottamooli_single_remedies?.length > 0 && (
              <Card title="🌿 Ottamooli (Single Remedies)">
                {treatment.ottamooli_single_remedies.map((r: any, i: number) => (
                  <div key={i} style={{ marginBottom: '1rem', padding: '1rem', background: '#f0f7e8', borderRadius: '0.75rem', borderLeft: '4px solid #4a7c28' }}>
                    <div style={{ fontWeight: 700, fontSize: '1.125rem', color: '#2d5016' }}>{r.medicine_name}</div>
                    <div style={{ fontStyle: 'italic', color: '#5a9632', marginBottom: '0.75rem' }}>{r.sanskrit_name}</div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.875rem' }}>
                      <div><strong>Part:</strong> {r.part_used}</div>
                      <div><strong>Dosage:</strong> {r.dosage}</div>
                      <div><strong>Preparation:</strong> {r.preparation}</div>
                      <div><strong>Timing:</strong> {r.timing}</div>
                      <div style={{ gridColumn: '1 / -1' }}><strong>Duration:</strong> {r.duration}</div>
                    </div>
                  </div>
                ))}
              </Card>
            )}

            {/* Classical Formulations */}
            {treatment.classical_formulations?.length > 0 && (
              <Card title="📜 Classical Formulations">
                {treatment.classical_formulations.map((f: any, i: number) => (
                  <div key={i} style={{ marginBottom: '0.75rem', padding: '1rem', background: '#fff8e1', borderRadius: '0.75rem' }}>
                    <div style={{ fontWeight: 700 }}>{f.name}</div>
                    <div style={{ fontSize: '0.875rem', color: '#666' }}>{f.english_name}</div>
                    <div style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
                      <strong>Form:</strong> {f.form} | <strong>Dosage:</strong> {f.dosage}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#f57c00', marginTop: '0.25rem' }}>📚 {f.reference_text}</div>
                  </div>
                ))}
              </Card>
            )}

            {/* Dietary Advice */}
            {treatment.pathya_dietary_advice && (
              <Card title="🍽️ Pathya (Dietary Advice)">
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <div>
                    <h4 style={{ color: '#4a7c28', marginBottom: '0.5rem' }}>✓ Foods to Favor</h4>
                    <ul style={{ fontSize: '0.875rem' }}>
                      {treatment.pathya_dietary_advice.foods_to_favor?.map((f: string, i: number) => <li key={i}>{f}</li>)}
                    </ul>
                  </div>
                  <div>
                    <h4 style={{ color: '#d32f2f', marginBottom: '0.5rem' }}>✗ Foods to Avoid</h4>
                    <ul style={{ fontSize: '0.875rem' }}>
                      {treatment.pathya_dietary_advice.foods_to_avoid?.map((f: string, i: number) => <li key={i}>{f}</li>)}
                    </ul>
                  </div>
                </div>
                {treatment.pathya_dietary_advice.specific_dietary_rules && (
                  <div style={{ marginTop: '1rem', padding: '1rem', background: '#fff8e1', borderRadius: '0.5rem', fontSize: '0.875rem' }}>
                    💡 {treatment.pathya_dietary_advice.specific_dietary_rules}
                  </div>
                )}
              </Card>
            )}

            {/* Lifestyle & Yoga */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              {treatment.vihara_lifestyle?.length > 0 && (
                <Card title="🏃 Vihara (Lifestyle)">
                  <ul style={{ fontSize: '0.875rem', lineHeight: 1.6 }}>
                    {treatment.vihara_lifestyle.map((v: string, i: number) => <li key={i}>{v}</li>)}
                  </ul>
                </Card>
              )}
              {treatment.yoga_exercises?.length > 0 && (
                <Card title="🧘 Yoga Exercises">
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {treatment.yoga_exercises.map((y: string, i: number) => (
                      <span key={i} style={{ padding: '0.5rem 1rem', background: '#e8f5e9', borderRadius: '2rem', fontSize: '0.875rem' }}>
                        {y}
                      </span>
                    ))}
                  </div>
                </Card>
              )}
            </div>

            {/* Warning Signs */}
            {treatment.warning_signs?.length > 0 && (
              <Card title="⚠️ Warning Signs">
                {treatment.warning_signs.map((w: string, i: number) => (
                  <div key={i} style={{ padding: '0.75rem', background: '#ffebee', border: '1px solid #ef5350', borderRadius: '0.5rem', marginBottom: '0.5rem', color: '#c62828', fontSize: '0.875rem' }}>
                    ! {w}
                  </div>
                ))}
              </Card>
            )}

            {/* Disclaimer */}
            <div style={{ padding: '1.5rem', background: '#f5f5f5', borderRadius: '1rem', textAlign: 'center', fontSize: '0.875rem', color: '#666' }}>
              ⚕️ {treatment.disclaimer}
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </main>
  )
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ background: 'white', borderRadius: '1rem', padding: '1.5rem', boxShadow: '0 2px 8px rgba(0,0,0,0.08)' }}>
      <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem', color: '#2d5016' }}>{title}</h3>
      {children}
    </div>
  )
}
