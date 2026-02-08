'use client'

import { useState } from 'react'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'YOUR_MODAL_URL_HERE'

export default function Home() {
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<any>(null)
  const [error, setError] = useState('')

  const exampleSymptoms = ['headache', 'stomach pain', 'fever', 'loss of appetite', 'cough']

  const analyze = async () => {
    if (!input.trim()) return

    setLoading(true)
    setError('')
    setResults(null)

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 120000) // 2 min

    try {
      const response = await fetch(API_BASE, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: input }),
        signal: controller.signal
      })

      clearTimeout(timeout)

      if (!response.ok) throw new Error('Request failed')

      const data = await response.json()
      setResults(data)
    } catch (err: any) {
      setError(err.name === 'AbortError' ? 'Request timeout. The AI is processing. Please try again.' : 'NetworkError when attempting to fetch resource.')
    } finally {
      setLoading(false)
    }
  }

  const treatment = results?.results?.[0]?.treatment_info

  return (
    <main className="min-h-screen bg-gradient-to-b from-green-50 to-white p-4 sm:p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="text-sm text-green-700 font-medium mb-2">🌿 WHO ITA · SNOMED CT</div>
          <h1 className="text-4xl sm:text-5xl font-bold text-green-800 mb-2">AyurAssist</h1>
          <p className="text-gray-600">AI-Powered Ayurveda Clinical Decision Support</p>
        </div>

        {/* Input Section */}
        <div className="bg-white rounded-2xl shadow-lg p-6 sm:p-8 mb-6">
          <h2 className="text-2xl font-semibold text-gray-800 mb-2">Describe your symptoms</h2>
          <p className="text-gray-600 text-sm mb-4">Enter symptoms — AI will analyze with Bio_ClinicalBERT & AyurParam</p>

          <div className="flex flex-wrap gap-2 mb-4">
            {exampleSymptoms.map((symptom) => (
              <button
                key={symptom}
                onClick={() => setInput(symptom)}
                className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-full text-sm text-gray-700 transition"
              >
                {symptom}
              </button>
            ))}
          </div>

          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && analyze()}
              placeholder="e.g., I have severe headache and nausea"
              className="flex-1 px-4 py-3 border-2 border-gray-200 rounded-xl focus:outline-none focus:border-green-500"
            />
            <button
              onClick={analyze}
              disabled={loading || !input.trim()}
              className="px-8 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white font-semibold rounded-xl transition"
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
          </div>
        </div>

        {/* Loading */}
        {loading && (
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-6 text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-600 border-t-transparent mb-3"></div>
            <p className="text-blue-800">AI is analyzing... This may take 30-60s</p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4">
            <p className="text-red-700 font-medium">⚠️ Error</p>
            <p className="text-red-600">{error}</p>
          </div>
        )}

        {/* Results */}
        {results && treatment && (
          <div className="space-y-6">
            {/* Processing Time */}
            {results.processing_time && (
              <div className="text-center text-sm text-gray-600">
                ⏱️ Processed in {results.processing_time}
              </div>
            )}

            {/* Clinical Info */}
            <div className="grid sm:grid-cols-2 gap-4">
              <div className="bg-blue-50 rounded-xl p-5">
                <h3 className="text-sm font-bold text-blue-900 mb-3">CLINICAL ENTITIES</h3>
                <div className="flex flex-wrap gap-2">
                  {results.clinical_entities?.map((entity: any, i: number) => (
                    <span key={i} className="px-3 py-1 bg-white border border-blue-200 rounded-lg text-blue-800 text-sm">
                      {entity.word}
                    </span>
                  ))}
                </div>
              </div>

              <div className="bg-purple-50 rounded-xl p-5">
                <h3 className="text-sm font-bold text-purple-900 mb-3">MEDICAL CODES</h3>
                <div className="space-y-2 text-sm">
                  <div><span className="font-semibold">UMLS:</span> <span className="text-purple-700">{results.umls_cui || 'N/A'}</span></div>
                  <div><span className="font-semibold">SNOMED:</span> <span className="text-purple-700">{results.snomed_code || results.results?.[0]?.snomed_code || 'N/A'}</span></div>
                </div>
              </div>
            </div>

            {/* Main Condition */}
            <div className="bg-green-600 text-white rounded-xl p-6">
              <div className="flex justify-between items-start">
                <div>
                  <h2 className="text-3xl font-bold mb-1">{treatment.condition_name}</h2>
                  <p className="text-2xl text-green-100 italic">{treatment.sanskrit_name}</p>
                </div>
                <div className="text-right">
                  <div className="text-5xl font-bold">{results.results?.[0]?.match_score || '100'}%</div>
                  <div className="text-sm text-green-100">{results.results?.[0]?.match_type}</div>
                </div>
              </div>
            </div>

            {/* Clinical Overview */}
            <div className="bg-white rounded-xl shadow p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center text-2xl">📋</div>
                <h3 className="text-2xl font-bold text-gray-800">Clinical Overview</h3>
              </div>
              <p className="text-gray-700 leading-relaxed mb-4">{treatment.brief_description}</p>
              <div className="grid sm:grid-cols-2 gap-4 text-sm">
                <div className="p-3 bg-gray-50 rounded-lg">
                  <span className="font-semibold">Dosha Involvement:</span> {treatment.dosha_involvement}
                </div>
                <div className="p-3 bg-gray-50 rounded-lg">
                  <span className="font-semibold">Prognosis:</span> {treatment.prognosis}
                </div>
              </div>
            </div>

            {/* Nidana (Causes) */}
            {treatment.nidana_causes && treatment.nidana_causes.length > 0 && (
              <div className="bg-white rounded-xl shadow p-6">
                <h3 className="text-xl font-bold text-gray-800 mb-3">🔍 Nidana (Causes)</h3>
                <ul className="space-y-2">
                  {treatment.nidana_causes.map((cause: string, i: number) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-green-600 mt-1">•</span>
                      <span className="text-gray-700">{cause}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Symptoms */}
            {treatment.rupa_symptoms && treatment.rupa_symptoms.length > 0 && (
              <div className="bg-white rounded-xl shadow p-6">
                <h3 className="text-xl font-bold text-gray-800 mb-3">📝 Rupa (Symptoms)</h3>
                <div className="grid sm:grid-cols-2 gap-2">
                  {treatment.rupa_symptoms.map((symptom: string, i: number) => (
                    <div key={i} className="flex items-center gap-2 p-2 bg-gray-50 rounded">
                      <span className="text-green-600">✓</span>
                      <span className="text-gray-700">{symptom}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Single Remedies */}
            {treatment.ottamooli_single_remedies && treatment.ottamooli_single_remedies.length > 0 && (
              <div className="bg-white rounded-xl shadow p-6">
                <h3 className="text-xl font-bold text-gray-800 mb-4">🌿 Ottamooli (Single Remedies)</h3>
                <div className="space-y-4">
                  {treatment.ottamooli_single_remedies.map((remedy: any, i: number) => (
                    <div key={i} className="border-l-4 border-green-500 pl-4 py-2 bg-green-50">
                      <div className="font-bold text-lg text-gray-800">{remedy.medicine_name}</div>
                      <div className="text-green-700 italic mb-2">{remedy.sanskrit_name}</div>
                      <div className="grid sm:grid-cols-2 gap-2 text-sm">
                        <div><span className="font-semibold">Part used:</span> {remedy.part_used}</div>
                        <div><span className="font-semibold">Dosage:</span> {remedy.dosage}</div>
                        <div><span className="font-semibold">Preparation:</span> {remedy.preparation}</div>
                        <div><span className="font-semibold">Timing:</span> {remedy.timing}</div>
                        <div className="sm:col-span-2"><span className="font-semibold">Duration:</span> {remedy.duration}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Classical Formulations */}
            {treatment.classical_formulations && treatment.classical_formulations.length > 0 && (
              <div className="bg-white rounded-xl shadow p-6">
                <h3 className="text-xl font-bold text-gray-800 mb-4">📜 Classical Formulations</h3>
                <div className="space-y-3">
                  {treatment.classical_formulations.map((formula: any, i: number) => (
                    <div key={i} className="p-4 bg-amber-50 border border-amber-200 rounded-lg">
                      <div className="font-bold text-gray-800">{formula.name}</div>
                      <div className="text-sm text-gray-600 mb-2">{formula.english_name}</div>
                      <div className="text-sm space-y-1">
                        <div><span className="font-semibold">Form:</span> {formula.form}</div>
                        <div><span className="font-semibold">Dosage:</span> {formula.dosage}</div>
                        <div><span className="font-semibold">Reference:</span> <span className="italic">{formula.reference_text}</span></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Dietary Advice */}
            {treatment.pathya_dietary_advice && (
              <div className="bg-white rounded-xl shadow p-6">
                <h3 className="text-xl font-bold text-gray-800 mb-4">🍽️ Pathya (Dietary Advice)</h3>
                <div className="grid sm:grid-cols-2 gap-4 mb-4">
                  <div>
                    <h4 className="font-semibold text-green-700 mb-2">Foods to Favor ✓</h4>
                    <ul className="space-y-1">
                      {treatment.pathya_dietary_advice.foods_to_favor?.map((food: string, i: number) => (
                        <li key={i} className="text-sm text-gray-700 flex items-center gap-2">
                          <span className="text-green-600">+</span> {food}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold text-red-700 mb-2">Foods to Avoid ✗</h4>
                    <ul className="space-y-1">
                      {treatment.pathya_dietary_advice.foods_to_avoid?.map((food: string, i: number) => (
                        <li key={i} className="text-sm text-gray-700 flex items-center gap-2">
                          <span className="text-red-600">−</span> {food}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
                {treatment.pathya_dietary_advice.specific_dietary_rules && (
                  <div className="p-3 bg-blue-50 border border-blue-200 rounded text-sm">
                    <span className="font-semibold">Dietary Rules:</span> {treatment.pathya_dietary_advice.specific_dietary_rules}
                  </div>
                )}
              </div>
            )}

            {/* Lifestyle & Yoga */}
            <div className="grid sm:grid-cols-2 gap-6">
              {treatment.vihara_lifestyle && treatment.vihara_lifestyle.length > 0 && (
                <div className="bg-white rounded-xl shadow p-6">
                  <h3 className="text-xl font-bold text-gray-800 mb-3">🏃 Vihara (Lifestyle)</h3>
                  <ul className="space-y-2">
                    {treatment.vihara_lifestyle.map((advice: string, i: number) => (
                      <li key={i} className="flex items-start gap-2 text-sm">
                        <span className="text-blue-600 mt-1">→</span>
                        <span className="text-gray-700">{advice}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {treatment.yoga_exercises && treatment.yoga_exercises.length > 0 && (
                <div className="bg-white rounded-xl shadow p-6">
                  <h3 className="text-xl font-bold text-gray-800 mb-3">🧘 Yoga Exercises</h3>
                  <ul className="space-y-2">
                    {treatment.yoga_exercises.map((exercise: string, i: number) => (
                      <li key={i} className="flex items-start gap-2 text-sm">
                        <span className="text-purple-600 mt-1">✦</span>
                        <span className="text-gray-700">{exercise}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* Warning Signs */}
            {treatment.warning_signs && treatment.warning_signs.length > 0 && (
              <div className="bg-red-50 border-2 border-red-300 rounded-xl p-6">
                <h3 className="text-xl font-bold text-red-800 mb-3">⚠️ Warning Signs</h3>
                <ul className="space-y-2">
                  {treatment.warning_signs.map((sign: string, i: number) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-red-600 font-bold mt-1">!</span>
                      <span className="text-red-800">{sign}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Disclaimer */}
            {treatment.disclaimer && (
              <div className="bg-gray-100 border border-gray-300 rounded-xl p-5 text-sm text-gray-700">
                <span className="font-semibold">Disclaimer:</span> {treatment.disclaimer}
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  )
}
