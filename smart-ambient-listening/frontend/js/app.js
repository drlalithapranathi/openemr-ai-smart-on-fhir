/**
 * SMART Ambient Listening Application
 */

document.addEventListener('DOMContentLoaded', async () => {
    // State
    let smartClient = null;
    let patient = null;
    let recorder = null;
    let transcriptionHistory = [];

    // Configuration
    const config = window.SMART_CONFIG || {
        transcriptionServiceUrl: 'http://localhost:8001'
    };

    // DOM Elements
    const elements = {
        patientBanner: document.getElementById('patient-banner'),
        patientName: document.getElementById('patient-name'),
        patientDob: document.getElementById('patient-dob'),
        patientMrn: document.getElementById('patient-mrn'),
        connectionStatus: document.getElementById('connection-status'),
        statusText: document.getElementById('status-text'),
        btnAmbient: document.getElementById('btn-ambient'),
        recordingStatus: document.getElementById('recording-status'),
        transcriptionLoading: document.getElementById('transcription-loading'),
        transcriptionResults: document.getElementById('transcription-results'),
        transcriptionHistory: document.getElementById('transcription-history'),
        historyList: document.getElementById('history-list'),
        actionsSection: document.getElementById('actions-section'),
        btnCopy: document.getElementById('btn-copy'),
        btnClear: document.getElementById('btn-clear'),
        btnSummarize: document.getElementById('btn-summarize'),
        soapSection: document.getElementById('soap-section'),
        soapLoading: document.getElementById('soap-loading'),
        soapResults: document.getElementById('soap-results'),
        errorModal: document.getElementById('error-modal'),
        errorMessage: document.getElementById('error-message'),
        btnCloseError: document.getElementById('btn-close-error'),
        permissionModal: document.getElementById('permission-modal'),
        btnRequestPermission: document.getElementById('btn-request-permission'),
        visualizer: document.getElementById('visualizer')
    };

    /**
     * Initialize the application
     */
    async function init() {
        if (!AmbientRecorder.isSupported()) {
            showError('Your browser does not support audio recording.');
            return;
        }

        // Try to complete SMART authorization
        try {
            smartClient = await FHIR.oauth2.ready();
            await connectToEHR();
        } catch (error) {
            console.log('No SMART context available:', error.message);
            updateConnectionStatus(false, 'Not connected to EHR. Launch from OpenEMR patient dashboard.');
            // Allow standalone testing without SMART context
            elements.btnAmbient.disabled = false;
        }

        initRecorder();
        setupEventListeners();
    }

    /**
     * Connect to EHR and load patient data
     */
    async function connectToEHR() {
        try {
            updateConnectionStatus(true, 'Connected to OpenEMR');

            // Get patient from SMART context
            patient = await smartClient.patient.read();
            displayPatientBanner(patient);

            elements.btnAmbient.disabled = false;
        } catch (error) {
            console.error('EHR connection error:', error);
            updateConnectionStatus(false, 'Failed to load patient data');
        }
    }

    /**
     * Display patient information banner
     */
    function displayPatientBanner(patient) {
        const name = getPatientName(patient);
        const dob = patient.birthDate || '';
        const mrn = getPatientMRN(patient);

        elements.patientName.textContent = name;
        elements.patientDob.textContent = dob ? `DOB: ${dob}` : '';
        elements.patientMrn.textContent = mrn ? `MRN: ${mrn}` : '';
        elements.patientBanner.classList.remove('hidden');
    }

    /**
     * Get patient display name from FHIR Patient resource
     */
    function getPatientName(patient) {
        const name = patient.name?.[0];
        if (!name) return 'Unknown';
        const given = name.given?.join(' ') || '';
        const family = name.family || '';
        return `${given} ${family}`.trim();
    }

    /**
     * Get patient MRN from FHIR Patient resource
     */
    function getPatientMRN(patient) {
        if (!patient.identifier) return null;
        const mrnId = patient.identifier.find(
            id => id.type?.coding?.some(c => c.code === 'MR')
        );
        return mrnId?.value || patient.identifier[0]?.value || null;
    }

    /**
     * Update connection status indicator
     */
    function updateConnectionStatus(connected, message) {
        const indicator = elements.connectionStatus.querySelector('.status-indicator');
        indicator.classList.toggle('connected', connected);
        indicator.classList.toggle('disconnected', !connected);
        elements.statusText.textContent = message;
    }

    /**
     * Initialize audio recorder
     */
    function initRecorder() {
        recorder = new AmbientRecorder({
            visualizer: elements.visualizer,
            onStart: handleRecordingStart,
            onStop: handleRecordingStop,
            onError: handleRecordingError
        });
    }

    /**
     * Setup event listeners
     */
    function setupEventListeners() {
        elements.btnAmbient.addEventListener('click', toggleRecording);
        elements.btnCopy.addEventListener('click', copyTranscription);
        elements.btnClear.addEventListener('click', clearSession);
        elements.btnSummarize.addEventListener('click', generateSOAPNote);
        elements.btnCloseError.addEventListener('click', () => {
            elements.errorModal.classList.add('hidden');
        });

        elements.btnRequestPermission.addEventListener('click', async () => {
            elements.permissionModal.classList.add('hidden');
            const result = await recorder.requestPermission();
            if (result.granted) {
                toggleRecording();
            } else {
                showError('Microphone permission denied.');
            }
        });
    }

    /**
     * Toggle recording state
     */
    async function toggleRecording() {
        if (recorder.isRecording) {
            recorder.stop();
        } else {
            try {
                const permissionStatus = await navigator.permissions.query({ name: 'microphone' });
                if (permissionStatus.state === 'denied') {
                    showError('Microphone access is blocked. Enable it in browser settings.');
                    return;
                }
                if (permissionStatus.state === 'prompt') {
                    elements.permissionModal.classList.remove('hidden');
                    return;
                }
            } catch (e) {
                // permissions API not supported
            }
            await recorder.start();
        }
    }

    /**
     * Handle recording start
     */
    function handleRecordingStart() {
        elements.btnAmbient.classList.add('recording');
        elements.btnAmbient.querySelector('.btn-text').textContent = 'Stop Listening';
        elements.recordingStatus.classList.remove('hidden');
        elements.transcriptionResults.innerHTML = '';
    }

    /**
     * Handle recording stop
     */
    async function handleRecordingStop(blob) {
        elements.btnAmbient.classList.remove('recording');
        elements.btnAmbient.querySelector('.btn-text').textContent = 'Start Ambient Listening';
        elements.recordingStatus.classList.add('hidden');
        elements.transcriptionLoading.classList.remove('hidden');
        try {
            const result = await transcribeAudio(blob);
            displayTranscription(result);
            addToHistory(result);
            elements.actionsSection.classList.remove('hidden');
        } catch (error) {
            console.error('Transcription error:', error);
            showError(`Transcription failed: ${error.message}`);
        } finally {
            elements.transcriptionLoading.classList.add('hidden');
        }
    }

    /**
     * Handle recording error
     */
    function handleRecordingError(error) {
        showError(`Recording error: ${error.message}`);
        elements.btnAmbient.classList.remove('recording');
        elements.btnAmbient.querySelector('.btn-text').textContent = 'Start Ambient Listening';
        elements.recordingStatus.classList.add('hidden');
    }

    /**
     * Send audio to transcription service
     */
    async function transcribeAudio(blob) {
        const formData = new FormData();
        formData.append('audio', blob, 'recording.webm');

        if (patient) {
            formData.append('patient_id', patient.id);
        }

        const response = await fetch(`${config.transcriptionServiceUrl}/transcribe`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(error.detail || 'Transcription failed');
        }

        return response.json();
    }

    /**
     * Display transcription result
     */
    function displayTranscription(result) {
        const text = result.text || 'No transcription available';
        const duration = result.duration ? `${result.duration.toFixed(1)}s` : '';

        elements.transcriptionResults.innerHTML = `
            <div class="transcription-text">${escapeHtml(text)}</div>
            ${duration ? `<div class="transcription-meta"><span>Duration: ${duration}</span></div>` : ''}
        `;
    }

    /**
     * Add transcription to history
     */
    function addToHistory(result) {
        transcriptionHistory.push({
            text: result.text,
            timestamp: new Date().toISOString(),
            duration: result.duration
        });
        updateHistoryDisplay();
    }

    /**
     * Update history display
     */
    function updateHistoryDisplay() {
        if (transcriptionHistory.length === 0) {
            elements.transcriptionHistory.classList.add('hidden');
            return;
        }

        elements.transcriptionHistory.classList.remove('hidden');
        elements.historyList.innerHTML = transcriptionHistory.map(item => `
            <div class="history-item">
                <div class="history-time">${formatTime(item.timestamp)}</div>
                <div class="history-text">${escapeHtml(item.text)}</div>
            </div>
        `).join('');
    }

    /**
     * Copy transcription to clipboard
     */
    async function copyTranscription() {
        const text = transcriptionHistory.map(h => h.text).join('\n\n');
        try {
            await navigator.clipboard.writeText(text);
            elements.btnCopy.textContent = 'Copied!';
            setTimeout(() => elements.btnCopy.textContent = 'Copy Transcription', 2000);
        } catch (error) {
            showError('Failed to copy to clipboard');
        }
    }

    async function generateSOAPNote() {
        const text = transcriptionHistory.map(h => h.text).join('\n\n');
        if (!text.trim()) {
            showError('No transcription available to summarize');
            return;
        }
        elements.soapSection.classList.remove('hidden');
        elements.soapLoading.classList.remove('hidden');
        elements.soapResults.innerHTML = '';
        try {
            const response = await fetch('/summarize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            const result = await response.json();
            displaySOAPNote(result);
        } catch (error) {
            console.error('SOAP generation error:', error);
            showError('Failed to generate SOAP note: ' + error.message);
            elements.soapSection.classList.add('hidden');
        } finally {
            elements.soapLoading.classList.add('hidden');
        }
    }

    function displaySOAPNote(result) {
        const sections = parseSOAPSections(result.soap_note || result.summary || '');
        elements.soapResults.innerHTML = `
            <div class="soap-note">
                <div class="soap-section">
                    <h4>Subjective</h4>
                    <p>${escapeHtml(sections.subjective)}</p>
                </div>
                <div class="soap-section">
                    <h4>Objective</h4>
                    <p>${escapeHtml(sections.objective)}</p>
                </div>
                <div class="soap-section">
                    <h4>Assessment</h4>
                    <p>${escapeHtml(sections.assessment)}</p>
                </div>
                <div class="soap-section">
                    <h4>Plan</h4>
                    <p>${escapeHtml(sections.plan)}</p>
                </div>
                <button class="btn btn-secondary" onclick="copySOAPNote()">Copy SOAP Note</button>
            </div>
        `;
    }

    function parseSOAPSections(text) {
        const sections = {
            subjective: '',
            objective: '',
            assessment: '',
            plan: ''
        };
        const patterns = {
            subjective: /(?:subjective|S)[:.]?\s*([\s\S]*?)(?=(?:objective|O)[:.]|$)/i,
            objective: /(?:objective|O)[:.]?\s*([\s\S]*?)(?=(?:assessment|A)[:.]|$)/i,
            assessment: /(?:assessment|A)[:.]?\s*([\s\S]*?)(?=(?:plan|P)[:.]|$)/i,
            plan: /(?:plan|P)[:.]?\s*([\s\S]*?)$/i
        };
        for (const [key, pattern] of Object.entries(patterns)) {
            const match = text.match(pattern);
            if (match && match[1]) {
                sections[key] = match[1].trim();
            }
        }
        if (!sections.subjective && !sections.objective) {
            sections.subjective = text;
        }
        return sections;
    }

    window.copySOAPNote = async function() {
        const soapText = elements.soapResults.innerText;
        try {
            await navigator.clipboard.writeText(soapText);
            alert('SOAP note copied to clipboard!');
        } catch (error) {
            showError('Failed to copy SOAP note');
        }
    };

    function clearSession() {
        transcriptionHistory = [];
        elements.transcriptionResults.innerHTML = `
            <p class="placeholder-text">
                Click "Start Ambient Listening" to begin capturing audio.
                The transcription will appear here.
            </p>
        `;
        elements.transcriptionHistory.classList.add('hidden');
        elements.historyList.innerHTML = '';
        elements.actionsSection.classList.add('hidden');
        elements.soapSection.classList.add('hidden');
        elements.soapResults.innerHTML = '';
    }

    /**
     * Show error modal
     */
    function showError(message) {
        elements.errorMessage.textContent = message;
        elements.errorModal.classList.remove('hidden');
    }

    /**
     * Escape HTML
     */
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Format timestamp
     */
    function formatTime(isoString) {
        return new Date(isoString).toLocaleTimeString();
    }

    // Initialize
    init();
});