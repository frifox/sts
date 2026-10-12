// Actions, feedbacks, presets and variables: Caption, Translate and Voice,
// each toggled (or set on/off) from a button that shows its current state.

const { combineRgb } = require('@companion-module/base')

const COLORS = {
	offBg: combineRgb(0, 0, 0), // black
	offText: combineRgb(200, 200, 200),
	onBg: combineRgb(0x46, 0x82, 0x59), // #468259
	onText: combineRgb(255, 255, 255),
	stoppingBg: combineRgb(200, 120, 20), // Voice: finishing the sentence being spoken, then off
	disconnectedBg: combineRgb(70, 20, 20),
	disconnectedText: combineRgb(150, 110, 110),
}

// The three toggles: the state key the module keeps, and sts's actions.
const STAGES = {
	caption: { label: 'Caption', on: 'start', off: 'stop' },
	translate: { label: 'Translate', on: 'translate_on', off: 'translate_off' },
	voice: { label: 'Voice', on: 'voice_on', off: 'voice_off' },
}

const MODE_CHOICES = [
	{ id: 'toggle', label: 'Toggle' },
	{ id: 'on', label: 'On' },
	{ id: 'off', label: 'Off' },
]

/** What to send for `mode` given the current state. */
function actionFor(stage, mode, state) {
	const def = STAGES[stage]
	if (mode === 'on') return def.on
	if (mode === 'off') return def.off
	// Toggle. Voice while it's finishing its sentence counts as on its way off:
	// toggling it turns it back on (as clicking Voice on the page does).
	if (stage === 'voice' && state.voiceStopping) return def.on
	return state[stage] ? def.off : def.on
}

function actionDefinitions(instance) {
	const actions = {}
	for (const [stage, def] of Object.entries(STAGES)) {
		actions[stage] = {
			name: `${def.label}: toggle / on / off`,
			options: [{ type: 'dropdown', id: 'mode', label: 'Mode', default: 'toggle', choices: MODE_CHOICES }],
			callback: (event) => {
				instance.connection.send(actionFor(stage, event.options.mode, instance.connection.state))
			},
		}
	}
	return actions
}

function feedbackDefinitions(instance) {
	const state = () => instance.connection.state
	return {
		stage_on: {
			type: 'boolean',
			name: 'Caption / Translate / Voice is on',
			description: 'Voice counts as off once Voice off was pressed, while it finishes its sentence (see "Voice is finishing")',
			defaultStyle: { bgcolor: COLORS.onBg, color: COLORS.onText },
			options: [
				{
					type: 'dropdown',
					id: 'stage',
					label: 'Which',
					default: 'caption',
					choices: Object.entries(STAGES).map(([id, def]) => ({ id, label: def.label })),
				},
			],
			callback: (feedback) => {
				const s = state()
				const stage = feedback.options.stage
				if (!s.connected) return false
				if (stage === 'voice') return s.voice && !s.voiceStopping
				return !!s[stage]
			},
		},
		voice_stopping: {
			type: 'boolean',
			name: 'Voice is finishing (off after the sentence being spoken)',
			defaultStyle: { bgcolor: COLORS.stoppingBg, color: COLORS.onText },
			options: [],
			callback: () => state().connected && state().voiceStopping,
		},
		disconnected: {
			type: 'boolean',
			name: 'Not connected to sts',
			defaultStyle: { bgcolor: COLORS.disconnectedBg, color: COLORS.disconnectedText },
			options: [],
			callback: () => !state().connected,
		},
	}
}

function presetDefinitions() {
	const presets = {}
	for (const [stage, def] of Object.entries(STAGES)) {
		const feedbacks = [
			{ feedbackId: 'stage_on', options: { stage }, style: { bgcolor: COLORS.onBg, color: COLORS.onText } },
		]
		if (stage === 'voice') {
			feedbacks.push({
				feedbackId: 'voice_stopping',
				options: {},
				style: { bgcolor: COLORS.stoppingBg, color: COLORS.onText },
			})
		}
		feedbacks.push({
			feedbackId: 'disconnected',
			options: {},
			style: { bgcolor: COLORS.disconnectedBg, color: COLORS.disconnectedText },
		})
		presets[`${stage}_toggle`] = {
			type: 'button',
			category: 'Toggles',
			name: `${def.label} on/off`,
			style: { text: def.label, size: '18', color: COLORS.offText, bgcolor: COLORS.offBg },
			steps: [{ down: [{ actionId: stage, options: { mode: 'toggle' } }], up: [] }],
			feedbacks,
		}
	}
	return presets
}

function variableDefinitions() {
	return [
		{ variableId: 'caption', name: 'Caption: on / off' },
		{ variableId: 'translate', name: 'Translate: on / off' },
		{ variableId: 'voice', name: 'Voice: on / off / finishing' },
		{ variableId: 'connection', name: 'Connection to sts: connected / disconnected' },
		{ variableId: 'stage', name: 'sts start-up stage ("Listening" once ready)' },
	]
}

function variableValues(state) {
	const onOff = (on) => (on ? 'on' : 'off')
	return {
		caption: onOff(state.caption),
		translate: onOff(state.translate),
		voice: state.voiceStopping ? 'finishing' : onOff(state.voice),
		connection: state.connected ? 'connected' : 'disconnected',
		stage: state.stage,
	}
}

module.exports = {
	STAGES,
	actionFor,
	actionDefinitions,
	feedbackDefinitions,
	presetDefinitions,
	variableDefinitions,
	variableValues,
}
