// Bitfocus Companion module for sts / Live Translate: buttons that toggle
// Caption, Translate and Voice, colored by their current state - kept in sync
// over a websocket to sts's /control endpoint, whichever side (Companion or
// the sts page) made the change.

const { InstanceBase, InstanceStatus, Regex, runEntrypoint } = require('@companion-module/base')
const { StsConnection } = require('./src/connection')
const defs = require('./src/definitions')

class StsInstance extends InstanceBase {
	async init(config) {
		this.config = config
		this.connection = new StsConnection({
			onState: (state) => this._stateChanged(state),
			onConnection: (connected, message) => {
				this.updateStatus(connected ? InstanceStatus.Ok : InstanceStatus.Connecting, message)
			},
			log: (level, text) => this.log(level, text),
		})
		this.setActionDefinitions(defs.actionDefinitions(this))
		this.setFeedbackDefinitions(defs.feedbackDefinitions(this))
		this.setPresetDefinitions(defs.presetDefinitions())
		this.setVariableDefinitions(defs.variableDefinitions())
		this._stateChanged(this.connection.state)
		this._connect()
	}

	async destroy() {
		this.connection?.disconnect()
	}

	async configUpdated(config) {
		this.config = config
		this._connect()
	}

	getConfigFields() {
		return [
			{
				type: 'static-text',
				id: 'info',
				width: 12,
				label: 'sts',
				value:
					"Connects to sts's remote-control websocket (ws://host:port/control). sts serves it on its caption " +
					"page port (80 by default) - from another computer, at the sts computer's address.",
			},
			{
				type: 'textinput',
				id: 'host',
				label: 'Host',
				width: 8,
				default: '127.0.0.1',
				regex: Regex.HOSTNAME,
			},
			{
				type: 'number',
				id: 'port',
				label: 'Port',
				width: 4,
				default: 80,
				min: 1,
				max: 65535,
			},
		]
	}

	_connect() {
		const host = (this.config?.host || '').trim()
		const port = Number(this.config?.port) || 80
		if (!host) {
			this.connection.disconnect()
			this.updateStatus(InstanceStatus.BadConfig, 'Set the host sts runs on')
			return
		}
		this.connection.connect(host, port)
	}

	_stateChanged(state) {
		this.setVariableValues(defs.variableValues(state))
		this.checkFeedbacks('stage_on', 'voice_stopping', 'voice_is', 'voice_preparing', 'disconnected')
	}
}

runEntrypoint(StsInstance, [])
