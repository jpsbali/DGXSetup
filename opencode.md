# opencode VS Code Extension
A Visual Studio Code extension that integrates opencode directly into your development workflow.

Prerequisites
This extension requires the opencode CLI to be installed on your system. Visit opencode.ai for installation instructions.

Features
Quick Launch: Use Cmd+Esc (Mac) or Ctrl+Esc (Windows/Linux) to open opencode in a split terminal view, or focus an existing terminal session if one is already running.
New Session: Use Cmd+Shift+Esc (Mac) or Ctrl+Shift+Esc (Windows/Linux) to start a new opencode terminal session, even if one is already open. You can also click the opencode button in the UI.
Context Awareness: Automatically share your current selection or tab with opencode.
File Reference Shortcuts: Use Cmd+Option+K (Mac) or Alt+Ctrl+K (Linux/Windows) to insert file references. For example, @File#L37-42.
Support
This is an early release. If you encounter issues or have feedback, please create an issue at https://github.com/sst/opencode/issues.

Development
code sdks/vscode - Open the sdks/vscode directory in VS Code. Do not open from repo root.
bun install - Run inside the sdks/vscode directory.
Press F5 to start debugging - This launches a new VS Code window with the extension loaded.
Making Changes
tsc and esbuild watchers run automatically during debugging (visible in the Terminal tab). Changes to the extension are automatically rebuilt in the background.

To test your changes:

In the debug VS Code window, press Cmd+Shift+P
Search for Developer: Reload Window
Reload to see your changes without restarting the debug session
