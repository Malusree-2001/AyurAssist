'use client'

import { useState } from 'react'

// ── Types matching the Python backend response ──────────────────
interface ClinicalEntity {
  word: string
  entity_group?: string
  score: number
  start?: number
  end?: number
}

interface OttamooliRemedy {
  medicine_name: string
  sanskrit_name: string
  part_used: string
  preparation: string
  dosage: string
  timing: string
  duration: string
}

interface ClassicalFormulation {
  name: string
  english_name: string
  form: string
  dosage: string
  reference_text: string
}

interface PanchakarmaTherapy {
  therapy_name: string
  description: string
  indication: string
}

interface DietaryAdvice {
  foods_to_favor: string[]
  foods_to_avoid: string[]
  specific_dietary_rules: string
}

interface TreatmentInfo {
  condition_name: string
  sanskrit_name: string
  brief_description: string
  dosha_involvement: string
  nidana_causes: string[]
  purvarupa_prodromal_symptoms: string[]
  rupa_symptoms: string[]
  ottamooli_single_remedies: OttamooliRemedy[]
  classical_formulations: ClassicalFormulation[]
  panchakarma_treatments: PanchakarmaTherapy[]
  pathya_dietary_advice: DietaryAdvice
  vihara_lifestyle: string[]
  yoga_exercises: string[]
  modern_correlation: string
  prognosis: string
  warning_signs: string[]
  disclaimer: string
}

interface ConditionResult {
  input_entity: string
  match_type: string
  match_score: number
  ita_id: string
  ayurveda_term: string
  sanskrit: string
  snomed_code: string
  snomed_name: string
  who_description: string
  treatment_info: TreatmentInfo
}

interface APIResponse {
  input_text: string
  clinical_entities: ClinicalEntity[]
  umls_cui: string
  conditions_matched: number
  results: ConditionResult[]
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://aravindkv28--ayurparam-service-fastapi-app.modal.run'

export default function Home() {
  const [inputText, setInputText] = useState('')
  const [results, setResults] = useState<ConditionResult[]>([])
  const [clinicalEntities, setClinicalEntities] = useState<ClinicalEntity[]>([])
  const [umlsCui, setUmlsCui] = useState<string>('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState('')

  const handleAnalyze = async () => {
    if (!inputText.trim()) {
      setError('Please enter symptoms')
      return
    }
    
    setIsAnalyzing(true)
    setError('')
    setResults([])
    setClinicalEntities([])

    try {
      console.log('🔍 Sending request to:', API_BASE)
      
      const res = await fetch(`${API_BASE}`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ text: inputText }),
      })

      console.log('📡 Response status:', res.status)

      if (!res.ok) {
        const errorText = await res.text()
        console.error('❌ API Error:', errorText)
        throw new Error(`API error: ${res.status} - ${errorText}`)
      }
      
      const data: APIResponse = await res.json()
      console.log('✅ Response data:', data)
      
      if (data.results && data.results.length > 0) {
        setResults(data.results)
        setClinicalEntities(data.clinical_entities || [])
        setUmlsCui(data.umls_cui || 'N/A')
      } else {
        setError('No matching Ayurvedic conditions found. Please try rephrasing your symptoms.')
      }
    } catch (err: any) {
      console.error("❌ Connection Error:", err)
      setError(`Failed to connect: ${err.message}. Please check if the Modal deployment is running.`)
      setResults([])
      setClinicalEntities([])
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') { 
      e.preventDefault()
      handleAnalyze()
    }
  }

  const hasValidTreatmentData = (info: TreatmentInfo) => {
    return info && 
           info.ottamooli_single_remedies && 
           info.ottamooli_single_remedies.length > 0
  }

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-cream)', position: 'relative', zIndex: 1 }}>
      <div style={{ maxWidth: 900, margin: '0 auto', padding: '24px 20px 60px' }}>

        {/* ─── HEADER ─── */}
        <header style={{ textAlign: 'center', padding: '48px 0 36px' }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            background: 'var(--green-light)', color: 'var(--green-deep)',
            fontSize: 11, fontWeight: 600, letterSpacing: 1.5,
            textTransform: 'uppercase' as const, padding: '6px 16px', 
            borderRadius: 100, marginBottom: 16
          }}>
            🌿 WHO ITA Standards · SNOMED CT Mapped
          </div>
          <h1 style={{
            fontSize: 44, fontWeight: 700,
            color: 'var(--green-deep)', letterSpacing: -1, lineHeight: 1.1
          }}>
            Ayur<span style={{ 
              color: 'var(--green-soft)', 
              fontWeight: 300, 
              fontStyle: 'italic' 
            }}>Assist</span>
          </h1>
          <p style={{ fontSize: 15, color: 'var(--text-light)', marginTop: 8 }}>
            AI-Powered Ayurveda Clinical Decision Support
          </p>
          <div style={{
            width: 60, height: 3, margin: '20px auto 0', borderRadius: 2,
            background: 'linear-gradient(90deg, var(--green-soft), var(--amber))'
          }} />
        </header>

        {/* ─── SEARCH ─── */}
        <section style={{
          background: 'var(--bg-card)', borderRadius: 28, padding: 32,
          boxShadow: '0 4px 16px rgba(0,0,0,0.08)', 
          border: '1px solid var(--border)', 
          marginBottom: 28
        }}>
          <p style={{ fontSize: 20, fontWeight: 500, marginBottom: 4 }}>
            Describe your symptoms
          </p>
          <p style={{ fontSize: 13, color: 'var(--text-light)', marginBottom: 16 }}>
            Enter symptoms in plain language — AI will identify medical entities
          </p>
          <div style={{ display: 'flex', gap: 10, alignItems: 'stretch' }}>
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="e.g., headache, stomach pain, loss of appetite..."
              disabled={isAnalyzing}
              style={{
                flex: 1, padding: '14px 20px', fontSize: 16,
                border: '2px solid var(--border)', borderRadius: 12,
                background: 'var(--bg-cream)', color: 'var(--text-dark)', 
                outline: 'none',
                opacity: isAnalyzing ? 0.6 : 1
              }}
            />
            <button
              onClick={handleAnalyze}
              disabled={!inputText.trim() || isAnalyzing}
              style={{
                padding: '14px 28px',
                background: isAnalyzing 
                  ? '#9CA3AF' 
                  : 'linear-gradient(135deg, var(--green-deep), var(--green-mid))',
                color: '#fff', border: 'none', borderRadius: 12, 
                fontSize: 15, fontWeight: 600,
                cursor: (!inputText.trim() || isAnalyzing) ? 'not-allowed' : 'pointer',
                opacity: (!inputText.trim() || isAnalyzing) ? 0.5 : 1,
                transition: 'all 0.3s'
              }}
            >
              {isAnalyzing ? "Analyzing..." : "Analyze"}
            </button>
          </div>
        </section>

        {/* ─── ERROR ─── */}
        {error && (
          <div style={{
            background: '#FEE2E2', border: '1px solid #FECACA', borderRadius: 12,
            padding: '14px 20px', marginBottom: 20, color: '#991B1B', fontSize: 14
          }}>
            ⚠️ {error}
          </div>
        )}

        {/* ─── LOADING STATE ─── */}
        {isAnalyzing && (
          <div style={{
            textAlign: 'center', padding: 40, background: 'var(--bg-card)',
            borderRadius: 20, border: '1px solid var(--border)'
          }}>
            <div style={{ fontSize: 40, marginBottom: 16 }}>🔬</div>
            <p style={{ fontSize: 16, color: 'var(--text-mid)' }}>
              Processing with Bio_ClinicalBERT and AyurParam...
            </p>
          </div>
        )}

        {/* ─── CLINICAL INTELLIGENCE DISPLAY ─── */}
        {!isAnalyzing && results.length > 0 && (
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: '1fr 1fr', 
            gap: 16, 
            marginBottom: 24 
          }}>
            <div style={{ 
              padding: 20, 
              background: '#EFF6FF', 
              border: '1px solid #DBEAFE', 
              borderRadius: 16 
            }}>
              <h3 style={{ 
                fontSize: 11, 
                fontWeight: 700, 
                color: '#1E40AF', 
                textTransform: 'uppercase', 
                marginBottom: 12 
              }}>
                Clinical Entities (Bio_ClinicalBERT)
              </h3>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {clinicalEntities.length > 0 ? (
                  clinicalEntities.map((ent, i) => (
                    <span 
                      key={i} 
                      style={{ 
                        padding: '4px 10px', 
                        background: '#fff', 
                        border: '1px solid #BFDBFE', 
                        borderRadius: 8, 
                        fontSize: 12, 
                        color: '#1D4ED8' 
                      }}
                    >
                      {ent.word}
                      {ent.score && (
                        <span style={{ fontSize: 10, opacity: 0.7, marginLeft: 4 }}>
                          ({Math.round(ent.score * 100)}%)
                        </span>
                      )}
                    </span>
                  ))
                ) : (
                  <span style={{ fontSize: 12, fontStyle: 'italic', color: '#60A5FA' }}>
                    No specific entities detected
                  </span>
                )}
              </div>
            </div>

            <div style={{ 
              padding: 20, 
              background: '#FAF5FF', 
              border: '1px solid #F3E8FF', 
              borderRadius: 16 
            }}>
              <h3 style={{ 
                fontSize: 11, 
                fontWeight: 700, 
                color: '#6B21A8', 
                textTransform: 'uppercase', 
                marginBottom: 12 
              }}>
                Standardized Mapping
              </h3>
              <div style={{ fontSize: 13, color: '#7E22CE', fontWeight: 500 }}>
                UMLS CUI: 
                <span style={{ 
                  fontFamily: 'monospace', 
                  background: '#fff', 
                  padding: '2px 6px', 
                  borderRadius: 4,
                  marginLeft: 6
                }}>
                  {umlsCui}
                </span>
              </div>
              <div style={{ 
                fontSize: 13, 
                color: '#7E22CE', 
                fontWeight: 500, 
                marginTop: 8 
              }}>
                SNOMED CT: 
                <span style={{ 
                  fontFamily: 'monospace', 
                  background: '#fff', 
                  padding: '2px 6px', 
                  borderRadius: 4,
                  marginLeft: 6
                }}>
                  {results[0]?.snomed_code || 'N/A'}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* ─── RESULTS ─── */}
        {!isAnalyzing && results.map((result, index) => {
          const info = result.treatment_info
          const hasLLM = hasValidTreatmentData(info)

          return (
            <div key={index} style={{ marginBottom: 32 }}>
              <div style={{
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'space-between',
                padding: '14px 20px', 
                background: 'var(--green-deep)', 
                borderRadius: 12, 
                marginBottom: 20, 
                color: '#fff'
              }}>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 600 }}>
                    Ayurvedic Protocol Generated
                  </div>
                  <div style={{ fontSize: 12, opacity: 0.8 }}>
                    Match Type: {result.match_type}
                  </div>
                </div>
                <div style={{ fontSize: 28, fontWeight: 700 }}>
                  {result.match_score}%
                </div>
              </div>

              <Card>
                <h2 style={{ fontSize: 30, color: 'var(--green-deep)' }}>
                  {info.condition_name || result.ayurveda_term}
                </h2>
                <p style={{ 
                  color: 'var(--terra)', 
                  fontStyle: 'italic', 
                  fontSize: 16,
                  marginTop: 8
                }}>
                  {info.sanskrit_name || result.sanskrit}
                </p>
                <div style={{ 
                  fontSize: 13, 
                  color: 'var(--text-light)', 
                  marginTop: 10,
                  padding: 10,
                  background: 'var(--bg-warm)',
                  borderRadius: 8
                }}>
                  <strong>SNOMED CT:</strong> {result.snomed_name} ({result.snomed_code})
                </div>
              </Card>

              <SectionCard icon="📋" iconBg="#DCFCE7" title="Clinical Description">
                <p style={{ fontSize: 15, lineHeight: 1.6, color: 'var(--text-mid)' }}>
                  {info.brief_description || result.who_description}
                </p>
                {info.dosha_involvement && (
                  <div style={{ 
                    marginTop: 12, 
                    padding: 10, 
                    background: '#FEF3C7',
                    borderRadius: 8
                  }}>
                    <strong>Dosha Involvement:</strong> {info.dosha_involvement}
                  </div>
                )}
              </SectionCard>

              {info.nidana_causes && info.nidana_causes.length > 0 && (
                <SectionCard icon="🔍" iconBg="#FEE2E2" title="Causes (Nidana)">
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {info.nidana_causes.map((cause, i) => (
                      <li key={i} style={{ marginBottom: 8, color: 'var(--text-mid)' }}>
                        {cause}
                      </li>
                    ))}
                  </ul>
                </SectionCard>
              )}

              {info.rupa_symptoms && info.rupa_symptoms.length > 0 && (
                <SectionCard icon="🩺" iconBg="#E0E7FF" title="Symptoms (Rupa)">
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {info.rupa_symptoms.map((symptom, i) => (
                      <li key={i} style={{ marginBottom: 8, color: 'var(--text-mid)' }}>
                        {symptom}
                      </li>
                    ))}
                  </ul>
                </SectionCard>
              )}

              {hasLLM && (
                <>
                  <SectionCard icon="🌿" iconBg="#DCFCE7" title="Ottamooli (Single Remedies)">
                    {info.ottamooli_single_remedies.map((remedy, i) => (
                      <div 
                        key={i} 
                        style={{ 
                          padding: 12, 
                          borderBottom: i !== info.ottamooli_single_remedies.length - 1 
                            ? '1px solid var(--border)' 
                            : 'none' 
                        }}
                      >
                        <strong style={{ color: 'var(--green-deep)', fontSize: 16 }}>
                          {remedy.medicine_name}
                        </strong>
                        <span style={{ color: 'var(--text-light)', marginLeft: 8 }}>
                          ({remedy.sanskrit_name})
                        </span>
                        <div style={{ 
                          fontSize: 13, 
                          color: 'var(--text-mid)', 
                          marginTop: 6,
                          lineHeight: 1.5
                        }}>
                          <div><strong>Part Used:</strong> {remedy.part_used}</div>
                          <div><strong>Preparation:</strong> {remedy.preparation}</div>
                          <div><strong>Dosage:</strong> {remedy.dosage}</div>
                          <div><strong>Timing:</strong> {remedy.timing}</div>
                          <div><strong>Duration:</strong> {remedy.duration}</div>
                        </div>
                      </div>
                    ))}
                  </SectionCard>

                  {info.classical_formulations && info.classical_formulations.length > 0 && (
                    <SectionCard icon="📚" iconBg="#FEF3C7" title="Classical Formulations">
                      {info.classical_formulations.map((formulation, i) => (
                        <div 
                          key={i} 
                          style={{ 
                            padding: 12, 
                            borderBottom: i !== info.classical_formulations.length - 1 
                              ? '1px solid var(--border)' 
                              : 'none' 
                          }}
                        >
                          <strong style={{ color: 'var(--amber)', fontSize: 16 }}>
                            {formulation.name}
                          </strong>
                          <span style={{ color: 'var(--text-light)', marginLeft: 8 }}>
                            ({formulation.english_name})
                          </span>
                          <div style={{ fontSize: 13, color: 'var(--text-mid)', marginTop: 6 }}>
                            <div><strong>Form:</strong> {formulation.form}</div>
                            <div><strong>Dosage:</strong> {formulation.dosage}</div>
                            <div><strong>Reference:</strong> {formulation.reference_text}</div>
                          </div>
                        </div>
                      ))}
                    </SectionCard>
                  )}

                  <SectionCard icon="🍽️" iconBg="#FEF3C7" title="Dietary Guidelines (Pathya-Apathya)">
                    <div style={{ marginBottom: 16 }}>
                      <strong style={{ fontSize: 14, color: '#166534' }}>
                        ✅ Foods to Favor (Pathya):
                      </strong>
                      <div style={{ 
                        fontSize: 14, 
                        color: '#166534', 
                        marginTop: 6,
                        paddingLeft: 20
                      }}>
                        {info.pathya_dietary_advice.foods_to_favor.join(', ')}
                      </div>
                    </div>
                    <div style={{ marginBottom: 16 }}>
                      <strong style={{ fontSize: 14, color: '#991B1B' }}>
                        ❌ Foods to Avoid (Apathya):
                      </strong>
                      <div style={{ 
                        fontSize: 14, 
                        color: '#991B1B', 
                        marginTop: 6,
                        paddingLeft: 20
                      }}>
                        {info.pathya_dietary_advice.foods_to_avoid.join(', ')}
                      </div>
                    </div>
                    <div style={{ 
                      marginTop: 12, 
                      fontSize: 13, 
                      fontStyle: 'italic', 
                      borderTop: '1px solid var(--border)', 
                      paddingTop: 12,
                      color: 'var(--text-mid)'
                    }}>
                      <strong>Note:</strong> {info.pathya_dietary_advice.specific_dietary_rules}
                    </div>
                  </SectionCard>

                  {info.vihara_lifestyle && info.vihara_lifestyle.length > 0 && (
                    <SectionCard icon="🧘" iconBg="#E0E7FF" title="Lifestyle (Vihara)">
                      <ul style={{ margin: 0, paddingLeft: 20 }}>
                        {info.vihara_lifestyle.map((item, i) => (
                          <li key={i} style={{ marginBottom: 8, color: 'var(--text-mid)' }}>
                            {item}
                          </li>
                        ))}
                      </ul>
                    </SectionCard>
                  )}

                  {info.yoga_exercises && info.yoga_exercises.length > 0 && (
                    <SectionCard icon="🧘‍♀️" iconBg="#FED7AA" title="Yoga Exercises">
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                        {info.yoga_exercises.map((exercise, i) => (
                          <span 
                            key={i}
                            style={{
                              padding: '6px 12px',
                              background: '#FFF7ED',
                              border: '1px solid #FDBA74',
                              borderRadius: 8,
                              fontSize: 13,
                              color: '#9A3412'
                            }}
                          >
                            {exercise}
                          </span>
                        ))}
                      </div>
                    </SectionCard>
                  )}
                </>
              )}

              {info.warning_signs && info.warning_signs.length > 0 && (
                <div style={{
                  background: '#FEE2E2',
                  border: '2px solid #FCA5A5',
                  borderRadius: 12,
                  padding: 16,
                  marginTop: 20
                }}>
                  <h3 style={{ 
                    fontSize: 15, 
                    fontWeight: 700, 
                    color: '#991B1B',
                    marginBottom: 12
                  }}>
                    ⚠️ Warning Signs - Seek Medical Attention If:
                  </h3>
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {info.warning_signs.map((sign, i) => (
                      <li key={i} style={{ 
                        marginBottom: 8, 
                        color: '#991B1B',
                        fontWeight: 500
                      }}>
                        {sign}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {info.disclaimer && (
                <div style={{
                  background: '#F3F4F6',
                  borderLeft: '4px solid #6B7280',
                  padding: 12,
                  marginTop: 20,
                  fontSize: 12,
                  color: '#4B5563',
                  fontStyle: 'italic'
                }}>
                  <strong>Disclaimer:</strong> {info.disclaimer}
                </div>
              )}
            </div>
          )
        })}

        {/* ─── FOOTER ─── */}
        {!isAnalyzing && results.length === 0 && !error && (
          <div style={{
            textAlign: 'center',
            padding: 40,
            color: 'var(--text-light)',
            fontSize: 14
          }}>
            Enter symptoms above to get AI-powered Ayurvedic recommendations
          </div>
        )}
      </div>
    </div>
  )
}

/* ─── Reusable UI Components ─── */
function Card({ children, style }: { 
  children: React.ReactNode
  style?: React.CSSProperties 
}) {
  return (
    <div style={{
      background: 'var(--bg-card)', 
      borderRadius: 20, 
      padding: 24,
      boxShadow: '0 1px 3px rgba(0,0,0,0.06)', 
      border: '1px solid var(--border)',
      marginBottom: 12, 
      ...style
    }}>
      {children}
    </div>
  )
}

function SectionCard({ 
  icon, 
  iconBg, 
  title, 
  children 
}: { 
  icon: string
  iconBg: string
  title: string
  children: React.ReactNode 
}) {
  return (
    <Card>
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: 10, 
        marginBottom: 16 
      }}>
        <div style={{ 
          width: 36, 
          height: 36, 
          borderRadius: 10, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center', 
          background: iconBg,
          fontSize: 18
        }}>
          {icon}
        </div>
        <span style={{ fontSize: 18, fontWeight: 600 }}>
          {title}
        </span>
      </div>
      {children}
    </Card>
  )
}
