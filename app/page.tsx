'use client'

import { useState } from 'react'

interface ClinicalEntity {
  word: string
  entity_group?: string
  score: number
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
  rupa_symptoms: string[]
  ottamooli_single_remedies: OttamooliRemedy[]
  classical_formulations: ClassicalFormulation[]
  pathya_dietary_advice: DietaryAdvice
  vihara_lifestyle: string[]
  yoga_exercises: string[]
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
  processing_time?: string
  results: ConditionResult[]
}

// IMPORTANT: Replace this with your actual Modal URL
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'YOUR_MODAL_URL_HERE'

export default function Home() {
  const [inputText, setInputText] = useState('')
  const [results, setResults] = useState<ConditionResult[]>([])
  const [clinicalEntities, setClinicalEntities] = useState<ClinicalEntity[]>([])
  const [umlsCui, setUmlsCui] = useState<string>('')
  const [processingTime, setProcessingTime] = useState<string>('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState('')

  const handleAnalyze = async () => {
    if (!inputText.trim()) {
      setError('Please enter your symptoms')
      return
    }
    
    setIsAnalyzing(true)
    setError('')
    setResults([])
    setClinicalEntities([])
    setProcessingTime('')

    try {
      console.log('🚀 Sending to:', API_BASE)
      console.log('📝 Input:', inputText)
      
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 120000) // 2 min timeout
      
      const res = await fetch(API_BASE, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ text: inputText }),
        signal: controller.signal
      })

      clearTimeout(timeoutId)
      
      console.log('📡 Status:', res.status)

      if (!res.ok) {
        const errorText = await res.text()
        console.error('❌ Error response:', errorText)
        throw new Error(`Server error (${res.status}). The AI model may be loading. Please try again in 30 seconds.`)
      }
      
      const data: APIResponse = await res.json()
      console.log('✅ Success:', data)
      
      if (data.results && data.results.length > 0) {
        setResults(data.results)
        setClinicalEntities(data.clinical_entities || [])
        setUmlsCui(data.umls_cui || 'N/A')
        setProcessingTime(data.processing_time || '')
      } else {
        setError('No Ayurvedic treatment found. Try different symptoms or rephrasing.')
      }
    } catch (err: any) {
      console.error("💥 Error:", err)
      
      if (err.name === 'AbortError') {
        setError('Request timeout. The AI is processing. Please try again.')
      } else if (err.message.includes('Failed to fetch')) {
        setError(`Cannot connect to AI service. Check: (1) Modal deployment is running, (2) Correct API_URL in .env.local: ${API_BASE}`)
      } else {
        setError(err.message || 'Connection failed. Please try again.')
      }
      
      setResults([])
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isAnalyzing) { 
      e.preventDefault()
      handleAnalyze()
    }
  }

  return (
    <div style={{ minHeight: '100vh', background: '#F5F3EF' }}>
      <div style={{ maxWidth: 900, margin: '0 auto', padding: '24px 20px 60px' }}>

        {/* HEADER */}
        <header style={{ textAlign: 'center', padding: '48px 0 36px' }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            background: '#E8F5E9', color: '#2E7D32',
            fontSize: 11, fontWeight: 600, letterSpacing: 1.5,
            textTransform: 'uppercase', padding: '6px 16px', 
            borderRadius: 100, marginBottom: 16
          }}>
            🌿 WHO ITA · SNOMED CT
          </div>
          <h1 style={{
            fontSize: 44, fontWeight: 700,
            background: 'linear-gradient(135deg, #2E7D32, #66BB6A)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            letterSpacing: -1, lineHeight: 1.1
          }}>
            AyurAssist
          </h1>
          <p style={{ fontSize: 15, color: '#666', marginTop: 8 }}>
            AI-Powered Ayurveda Clinical Decision Support
          </p>
        </header>

        {/* SEARCH */}
        <section style={{
          background: '#fff', borderRadius: 24, padding: 32,
          boxShadow: '0 2px 12px rgba(0,0,0,0.08)', 
          border: '1px solid #E0E0E0', 
          marginBottom: 24
        }}>
          <p style={{ fontSize: 20, fontWeight: 500, marginBottom: 4, color: '#333' }}>
            Describe your symptoms
          </p>
          <p style={{ fontSize: 13, color: '#666', marginBottom: 16 }}>
            Enter symptoms — AI will analyze with Bio_ClinicalBERT & AyurParam
          </p>
          
          {/* Example pills */}
          <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
            {['headache', 'stomach pain', 'fever', 'loss of appetite', 'cough'].map(ex => (
              <button
                key={ex}
                onClick={() => setInputText(ex)}
                disabled={isAnalyzing}
                style={{
                  padding: '4px 12px',
                  background: '#F5F5F5',
                  border: '1px solid #E0E0E0',
                  borderRadius: 16,
                  fontSize: 12,
                  color: '#666',
                  cursor: isAnalyzing ? 'not-allowed' : 'pointer',
                  opacity: isAnalyzing ? 0.5 : 1
                }}
              >
                {ex}
              </button>
            ))}
          </div>
          
          <div style={{ display: 'flex', gap: 10 }}>
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="e.g., headache, nausea, joint pain..."
              disabled={isAnalyzing}
              style={{
                flex: 1, padding: '14px 20px', fontSize: 16,
                border: '2px solid #E0E0E0', borderRadius: 12,
                background: '#FAFAFA', color: '#333', 
                outline: 'none',
                opacity: isAnalyzing ? 0.6 : 1
              }}
            />
            <button
              onClick={handleAnalyze}
              disabled={!inputText.trim() || isAnalyzing}
              style={{
                padding: '14px 32px',
                background: isAnalyzing 
                  ? '#9E9E9E' 
                  : 'linear-gradient(135deg, #2E7D32, #66BB6A)',
                color: '#fff', border: 'none', borderRadius: 12, 
                fontSize: 15, fontWeight: 600,
                cursor: (!inputText.trim() || isAnalyzing) ? 'not-allowed' : 'pointer',
                opacity: (!inputText.trim() || isAnalyzing) ? 0.6 : 1,
                transition: 'all 0.3s',
                minWidth: 120
              }}
            >
              {isAnalyzing ? "Analyzing..." : "Analyze"}
            </button>
          </div>
        </section>

        {/* ERROR */}
        {error && (
          <div style={{
            background: '#FFEBEE', border: '1px solid #FFCDD2', borderRadius: 12,
            padding: '16px 20px', marginBottom: 20, color: '#C62828', fontSize: 14,
            lineHeight: 1.6
          }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>⚠️ Error</div>
            {error}
          </div>
        )}

        {/* LOADING */}
        {isAnalyzing && (
          <div style={{
            textAlign: 'center', padding: 48, background: '#fff',
            borderRadius: 20, border: '1px solid #E0E0E0',
            marginBottom: 24
          }}>
            <div style={{ 
              width: 48, height: 48, margin: '0 auto 16px',
              border: '4px solid #E0E0E0',
              borderTop: '4px solid #2E7D32',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite'
            }} />
            <p style={{ fontSize: 16, color: '#666', marginBottom: 8 }}>
              Processing with AI models...
            </p>
            <p style={{ fontSize: 13, color: '#999' }}>
              This may take 15-60 seconds on first request
            </p>
            <style jsx>{`
              @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
              }
            `}</style>
          </div>
        )}

        {/* RESULTS */}
        {!isAnalyzing && results.length > 0 && (
          <>
            {/* Processing time */}
            {processingTime && (
              <div style={{ 
                textAlign: 'center', 
                fontSize: 12, 
                color: '#666',
                marginBottom: 16
              }}>
                ⏱️ Processed in {processingTime}
              </div>
            )}
            
            {/* Entity Cards */}
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: '1fr 1fr', 
              gap: 16, 
              marginBottom: 24 
            }}>
              <div style={{ 
                padding: 20, 
                background: '#E3F2FD', 
                border: '1px solid #BBDEFB', 
                borderRadius: 16 
              }}>
                <h3 style={{ 
                  fontSize: 11, 
                  fontWeight: 700, 
                  color: '#1565C0', 
                  textTransform: 'uppercase', 
                  marginBottom: 12,
                  letterSpacing: 1
                }}>
                  Clinical Entities
                </h3>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  {clinicalEntities.length > 0 ? (
                    clinicalEntities.map((ent, i) => (
                      <span 
                        key={i} 
                        style={{ 
                          padding: '6px 12px', 
                          background: '#fff', 
                          border: '1px solid #90CAF9', 
                          borderRadius: 8, 
                          fontSize: 13, 
                          color: '#1565C0',
                          fontWeight: 500
                        }}
                      >
                        {ent.word}
                      </span>
                    ))
                  ) : (
                    <span style={{ fontSize: 12, fontStyle: 'italic', color: '#64B5F6' }}>
                      General analysis
                    </span>
                  )}
                </div>
              </div>

              <div style={{ 
                padding: 20, 
                background: '#F3E5F5', 
                border: '1px solid #E1BEE7', 
                borderRadius: 16 
              }}>
                <h3 style={{ 
                  fontSize: 11, 
                  fontWeight: 700, 
                  color: '#6A1B9A', 
                  textTransform: 'uppercase', 
                  marginBottom: 12,
                  letterSpacing: 1
                }}>
                  Medical Codes
                </h3>
                <div style={{ fontSize: 13, color: '#7B1FA2', fontWeight: 500, marginBottom: 8 }}>
                  UMLS: <span style={{ 
                    fontFamily: 'monospace', 
                    background: '#fff', 
                    padding: '3px 8px', 
                    borderRadius: 4,
                    fontSize: 12
                  }}>
                    {umlsCui}
                  </span>
                </div>
                <div style={{ fontSize: 13, color: '#7B1FA2', fontWeight: 500 }}>
                  SNOMED: <span style={{ 
                    fontFamily: 'monospace', 
                    background: '#fff', 
                    padding: '3px 8px', 
                    borderRadius: 4,
                    fontSize: 12
                  }}>
                    {results[0]?.snomed_code}
                  </span>
                </div>
              </div>
            </div>

            {/* Treatment Results */}
            {results.map((result, index) => {
              const info = result.treatment_info
              const hasRemedies = info.ottamooli_single_remedies?.length > 0

              return (
                <div key={index}>
                  {/* Match Score */}
                  <div style={{
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'space-between',
                    padding: '16px 24px', 
                    background: 'linear-gradient(135deg, #2E7D32, #66BB6A)',
                    borderRadius: 12, 
                    marginBottom: 20, 
                    color: '#fff'
                  }}>
                    <div>
                      <div style={{ fontSize: 16, fontWeight: 600 }}>
                        Treatment Protocol
                      </div>
                      <div style={{ fontSize: 12, opacity: 0.9 }}>
                        {result.match_type}
                      </div>
                    </div>
                    <div style={{ fontSize: 32, fontWeight: 700 }}>
                      {result.match_score}%
                    </div>
                  </div>

                  {/* Condition Name */}
                  <Card>
                    <h2 style={{ fontSize: 28, color: '#2E7D32', marginBottom: 8 }}>
                      {info.condition_name}
                    </h2>
                    <p style={{ 
                      color: '#D84315', 
                      fontStyle: 'italic', 
                      fontSize: 16
                    }}>
                      {info.sanskrit_name}
                    </p>
                  </Card>

                  {/* Description */}
                  <SectionCard icon="📋" iconBg="#E8F5E9" title="Clinical Overview">
                    <p style={{ fontSize: 15, lineHeight: 1.7, color: '#424242' }}>
                      {info.brief_description}
                    </p>
                    {info.dosha_involvement && (
                      <div style={{ 
                        marginTop: 12, 
                        padding: 12, 
                        background: '#FFF8E1',
                        borderRadius: 8,
                        borderLeft: '3px solid #FFA726'
                      }}>
                        <strong style={{ color: '#E65100' }}>Dosha:</strong> {info.dosha_involvement}
                      </div>
                    )}
                  </SectionCard>

                  {/* Remedies */}
                  {hasRemedies && (
                    <SectionCard icon="🌿" iconBg="#E8F5E9" title="Single Herb Remedies (Ottamooli)">
                      {info.ottamooli_single_remedies.map((remedy, i) => (
                        <div 
                          key={i} 
                          style={{ 
                            padding: 14, 
                            background: i % 2 === 0 ? '#F9F9F9' : '#fff',
                            borderRadius: 8,
                            marginBottom: 10
                          }}
                        >
                          <strong style={{ color: '#2E7D32', fontSize: 15 }}>
                            {remedy.medicine_name}
                          </strong>
                          <span style={{ color: '#666', marginLeft: 8, fontSize: 13 }}>
                            ({remedy.sanskrit_name})
                          </span>
                          <div style={{ 
                            fontSize: 13, 
                            color: '#555', 
                            marginTop: 8,
                            display: 'grid',
                            gridTemplateColumns: '1fr 1fr',
                            gap: 8
                          }}>
                            <div>📌 <strong>Part:</strong> {remedy.part_used}</div>
                            <div>⚗️ <strong>Form:</strong> {remedy.preparation}</div>
                            <div>💊 <strong>Dose:</strong> {remedy.dosage}</div>
                            <div>⏰ <strong>When:</strong> {remedy.timing}</div>
                          </div>
                        </div>
                      ))}
                    </SectionCard>
                  )}

                  {/* Diet */}
                  {info.pathya_dietary_advice && (
                    <SectionCard icon="🍽️" iconBg="#FFF8E1" title="Dietary Guidelines">
                      <div style={{ marginBottom: 16 }}>
                        <div style={{ 
                          fontSize: 14, 
                          fontWeight: 600, 
                          color: '#2E7D32',
                          marginBottom: 8
                        }}>
                          ✅ Recommended (Pathya)
                        </div>
                        <div style={{ fontSize: 14, color: '#424242', lineHeight: 1.6 }}>
                          {info.pathya_dietary_advice.foods_to_favor?.join(', ') || 'Consult practitioner'}
                        </div>
                      </div>
                      <div>
                        <div style={{ 
                          fontSize: 14, 
                          fontWeight: 600, 
                          color: '#C62828',
                          marginBottom: 8
                        }}>
                          ❌ Avoid (Apathya)
                        </div>
                        <div style={{ fontSize: 14, color: '#424242', lineHeight: 1.6 }}>
                          {info.pathya_dietary_advice.foods_to_avoid?.join(', ') || 'Consult practitioner'}
                        </div>
                      </div>
                      {info.pathya_dietary_advice.specific_dietary_rules && (
                        <div style={{ 
                          marginTop: 14, 
                          fontSize: 13, 
                          fontStyle: 'italic',
                          color: '#666',
                          borderTop: '1px solid #E0E0E0',
                          paddingTop: 12
                        }}>
                          💡 {info.pathya_dietary_advice.specific_dietary_rules}
                        </div>
                      )}
                    </SectionCard>
                  )}

                  {/* Disclaimer */}
                  {info.disclaimer && (
                    <div style={{
                      background: '#FAFAFA',
                      borderLeft: '4px solid #9E9E9E',
                      padding: 14,
                      marginTop: 24,
                      fontSize: 12,
                      color: '#616161',
                      fontStyle: 'italic',
                      borderRadius: 4
                    }}>
                      <strong>⚠️ Disclaimer:</strong> {info.disclaimer}
                    </div>
                  )}
                </div>
              )
            })}
          </>
        )}

        {/* Empty State */}
        {!isAnalyzing && results.length === 0 && !error && (
          <div style={{
            textAlign: 'center',
            padding: 60,
            color: '#999',
            fontSize: 14
          }}>
            <div style={{ fontSize: 48, marginBottom: 16 }}>🌿</div>
            Enter your symptoms to receive AI-powered Ayurvedic guidance
          </div>
        )}
      </div>
    </div>
  )
}

function Card({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      background: '#fff', 
      borderRadius: 16, 
      padding: 24,
      boxShadow: '0 1px 3px rgba(0,0,0,0.08)', 
      border: '1px solid #E0E0E0',
      marginBottom: 16
    }}>
      {children}
    </div>
  )
}

function SectionCard({ icon, iconBg, title, children }: { 
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
        gap: 12, 
        marginBottom: 16 
      }}>
        <div style={{ 
          width: 40, 
          height: 40, 
          borderRadius: 10, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center', 
          background: iconBg,
          fontSize: 20
        }}>
          {icon}
        </div>
        <span style={{ fontSize: 18, fontWeight: 600, color: '#333' }}>
          {title}
        </span>
      </div>
      {children}
    </Card>
  )
}
