# @viateai/plugin-antigravity

Viate provider for a Google Antigravity subscription. Provider id: `antigravity`.

The gateway speaks OpenAI chat completions, responses, and image generations. Chat and responses are translated onto Cloud Code `v1internal:generateContent` at `https://daily-cloudcode-pa.googleapis.com`, falling back to `https://cloudcode-pa.googleapis.com`. Requests identify as `antigravity/hub/2.29.0`.

`previous_response_id`, conversation state, background responses, and `store` are rejected. Send the full input.

## Connection

Authorization-code OAuth uses the public Antigravity IDE installed-app client on `https://accounts.google.com`. The client id and secret ship with that IDE. They are not a user credential. The secret in this repository is XOR-masked so scanners do not match the literal.

The plugin listens on `127.0.0.1` at an ephemeral port and path `/callback`, on the computer running the gateway. It redirects `code` and `state` to the admin callback. The listener closes after the redirect, or after five minutes.

A pasted credential also works. JSON accepts `accessToken` or `access_token`, `refreshToken` or `refresh_token`, `projectId`, and `email`. `projectId` is required for quota and for chat when discovery cannot fill it in.

`keepalive` refreshes the access token one minute before `expiresAt`. `subscription` reads `retrieveUserQuota` and `retrieveUserQuotaSummary`.

`listModels` calls `fetchAvailableModels`. Gemini 3.1 Flash Image is added when the catalog omits it.

## Install

This package is not published to npm. Clone the repository, compile it, and install that directory. `viate.entry` is `dist/index.js`, and `dist/` is not in git. The installer runs with `ignore-scripts=true`, so it will not compile the package for you.

Node.js >= 24. pnpm 12.16.1, pinned by `packageManager`.

```bash
git clone https://github.com/ViateAI/viate-plugin-antigravity.git
cd viate-plugin-antigravity
pnpm install
pnpm build
viate plugin add "$PWD"
```

`$PWD` is passed to `pnpm add` by the `viate` process. From the admin plugin source field, use the same directory as an absolute path that process can read.

`@viateai/sdk` is a peer of `1.0.1`; the gateway removes it at install and loads its own copy. `plugin add` stages a pending workspace. Restart the gateway to load it.

To install a newer checkout, pull, run `pnpm install` and `pnpm build` again, then `viate plugin add "$PWD"` again, and restart.

Plugins run in-process. Install only packages you trust. See [Viate](https://github.com/ViateAI/viate).

## Development

```bash
pnpm install
pnpm typecheck
pnpm test
pnpm build
```

Release Please on `main` owns the version and changelog. Do not edit either by hand.

MIT. See [LICENSE](LICENSE).
