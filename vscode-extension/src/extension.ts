import * as vscode from "vscode";

/**
 * Read all git-review settings from VS Code configuration and return them as
 * an object of CLI argument fragments and environment variable overrides.
 */
function getConfig(): {
  githubToken: string;
  openaiApiKey: string;
  openaiBaseUrl: string;
  defaultRepo: string;
  defaultOwner: string;
  reviewAuthor: string;
  model: string;
  defaultDays: number;
  defaultBranch: string;
  enableThinking: boolean;
  sprintCapacity: number;
  sprints: number;
} {
  const cfg = vscode.workspace.getConfiguration("gitReview");
  return {
    githubToken: cfg.get<string>("githubToken", ""),
    openaiApiKey: cfg.get<string>("openaiApiKey", ""),
    openaiBaseUrl: cfg.get<string>("openaiBaseUrl", ""),
    defaultRepo: cfg.get<string>("defaultRepo", ""),
    defaultOwner: cfg.get<string>("defaultOwner", ""),
    reviewAuthor: cfg.get<string>("reviewAuthor", ""),
    model: cfg.get<string>("model", "gpt-4o-mini"),
    defaultDays: cfg.get<number>("defaultDays", 7),
    defaultBranch: cfg.get<string>("defaultBranch", ""),
    enableThinking: cfg.get<boolean>("enableThinking", false),
    sprintCapacity: cfg.get<number>("sprintCapacity", 10),
    sprints: cfg.get<number>("sprints", 3),
  };
}

/**
 * Build a shell-safe CLI argument string for the common authentication /
 * model flags shared by every git-review sub-command.
 */
function buildCommonArgs(cfg: ReturnType<typeof getConfig>): string[] {
  const args: string[] = [];
  if (cfg.githubToken) {
    args.push(`--token ${shellQuote(cfg.githubToken)}`);
  }
  if (cfg.openaiApiKey) {
    args.push(`--openai-key ${shellQuote(cfg.openaiApiKey)}`);
  }
  if (cfg.openaiBaseUrl) {
    args.push(`--base-url ${shellQuote(cfg.openaiBaseUrl)}`);
  }
  if (cfg.model) {
    args.push(`--model ${shellQuote(cfg.model)}`);
  }
  return args;
}

/**
 * Minimal shell quoting: wrap the value in single quotes and escape any
 * embedded single quotes so the resulting token is safe on POSIX shells and
 * PowerShell alike.
 */
function shellQuote(value: string): string {
  // Wrap in double-quotes and escape characters that are special inside them.
  return `"${value.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
}

/**
 * Send a command string to a dedicated git-review terminal, creating it if it
 * does not already exist, and make it visible.
 */
function runInTerminal(command: string): void {
  const existingTerminals = vscode.window.terminals;
  let terminal =
    existingTerminals.find((t: vscode.Terminal) => t.name === "git-review") ??
    vscode.window.createTerminal("git-review");
  terminal.show(/* preserveFocus */ false);
  terminal.sendText(command);
}

// ---------------------------------------------------------------------------
// Command: Review Repository
// ---------------------------------------------------------------------------

async function reviewRepository(): Promise<void> {
  const cfg = getConfig();

  let repo = cfg.defaultRepo.trim();
  let owner = cfg.defaultOwner.trim();

  if (!repo && !owner) {
    const input = await vscode.window.showInputBox({
      title: "Git Review: Review Repository",
      prompt:
        "Enter the GitHub repository to review (owner/repo), or leave blank to review by owner.",
      placeHolder: "owner/repo",
      validateInput: (value: string) => {
        if (!value) {
          return null; // blank → will ask for owner instead
        }
        if (!/^[^/]+\/[^/]+$/.test(value)) {
          return "Must be in 'owner/repo' format.";
        }
        return null;
      },
    });
    if (input === undefined) {
      return; // user cancelled
    }
    repo = input.trim();

    if (!repo) {
      const ownerInput = await vscode.window.showInputBox({
        title: "Git Review: Review Owner",
        prompt: "Enter the GitHub user or organization name to review all repos.",
        placeHolder: "my-org",
        validateInput: (value: string) => {
          if (!value || !value.trim()) {
            return "Owner is required.";
          }
          return null;
        },
      });
      if (!ownerInput || !ownerInput.trim()) {
        return; // user cancelled or empty
      }
      owner = ownerInput.trim();
    }
  }

  const args: string[] = ["git-review", "review"];

  if (repo) {
    args.push(`--repo ${shellQuote(repo)}`);
  } else {
    args.push(`--owner ${shellQuote(owner)}`);
  }

  if (cfg.reviewAuthor.trim()) {
    args.push(`--author ${shellQuote(cfg.reviewAuthor.trim())}`);
  }

  if (cfg.defaultBranch.trim()) {
    args.push(`--branch ${shellQuote(cfg.defaultBranch.trim())}`);
  }

  args.push(`--days ${cfg.defaultDays}`);
  args.push(...buildCommonArgs(cfg));

  runInTerminal(args.join(" "));
}

// ---------------------------------------------------------------------------
// Command: Generate Commit Message
// ---------------------------------------------------------------------------

async function generateCommitMessage(): Promise<void> {
  const cfg = getConfig();

  if (!cfg.openaiApiKey && !cfg.openaiBaseUrl) {
    const action = await vscode.window.showWarningMessage(
      "No OpenAI API key configured. Open settings to add one.",
      "Open Settings"
    );
    if (action === "Open Settings") {
      await vscode.commands.executeCommand(
        "workbench.action.openSettings",
        "gitReview.openaiApiKey"
      );
    }
    return;
  }

  // Resolve the repository path: prefer the first workspace folder.
  const workspaceFolder =
    vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? undefined;

  const args: string[] = ["git-review", "commit-message"];

  if (workspaceFolder) {
    args.push(`--repo-path ${shellQuote(workspaceFolder)}`);
  }

  if (cfg.enableThinking) {
    args.push("--think");
  }

  args.push(...buildCommonArgs(cfg));

  runInTerminal(args.join(" "));
}

// ---------------------------------------------------------------------------
// Command: Create Issues from Requirements
// ---------------------------------------------------------------------------

async function createIssues(): Promise<void> {
  const cfg = getConfig();

  if (!cfg.openaiApiKey && !cfg.openaiBaseUrl) {
    const action = await vscode.window.showWarningMessage(
      "No OpenAI API key configured. Open settings to add one.",
      "Open Settings"
    );
    if (action === "Open Settings") {
      await vscode.commands.executeCommand(
        "workbench.action.openSettings",
        "gitReview.openaiApiKey"
      );
    }
    return;
  }

  let repo = cfg.defaultRepo;
  if (!repo) {
    const input = await vscode.window.showInputBox({
      title: "Git Review: Create Issues",
      prompt: "Enter the target GitHub repository (owner/repo).",
      placeHolder: "owner/repo",
      validateInput: (value: string) => {
        if (!value || !value.trim()) {
          return "Repository is required.";
        }
        if (!/^[^/]+\/[^/]+$/.test(value.trim())) {
          return "Must be in 'owner/repo' format.";
        }
        return null;
      },
    });
    if (input === undefined) {
      return; // user cancelled
    }
    repo = input.trim();
  }

  // Let the user pick a requirements markdown file
  const uris = await vscode.window.showOpenDialog({
    title: "Select requirements markdown file",
    canSelectMany: false,
    filters: { Markdown: ["md", "markdown", "txt"] },
    openLabel: "Select Requirements File",
  });

  if (!uris || uris.length === 0) {
    return; // user cancelled
  }

  const requirementsFile = uris[0].fsPath;

  const args: string[] = [
    "git-review",
    "create-issues",
    `--repo ${shellQuote(repo)}`,
    `--requirements ${shellQuote(requirementsFile)}`,
  ];

  args.push(...buildCommonArgs(cfg));

  runInTerminal(args.join(" "));
}

// ---------------------------------------------------------------------------
// Command: Create Milestone
// ---------------------------------------------------------------------------

async function createMilestone(): Promise<void> {
  const cfg = getConfig();

  let repo = cfg.defaultRepo;
  if (!repo) {
    const input = await vscode.window.showInputBox({
      title: "Git Review: Create Milestone",
      prompt: "Enter the target GitHub repository (owner/repo).",
      placeHolder: "owner/repo",
      validateInput: (value: string) => {
        if (!value || !value.trim()) {
          return "Repository is required.";
        }
        if (!/^[^/]+\/[^/]+$/.test(value.trim())) {
          return "Must be in 'owner/repo' format.";
        }
        return null;
      },
    });
    if (input === undefined) {
      return;
    }
    repo = input.trim();
  }

  const title = await vscode.window.showInputBox({
    title: "Git Review: Create Milestone",
    prompt: "Enter the milestone title.",
    placeHolder: "Sprint 1",
    validateInput: (value: string) => {
      if (!value || !value.trim()) {
        return "Title is required.";
      }
      return null;
    },
  });
  if (!title || !title.trim()) {
    return;
  }

  const description = await vscode.window.showInputBox({
    title: "Git Review: Create Milestone — Description (optional)",
    prompt: "Enter an optional milestone description, or leave blank.",
    placeHolder: "Goals for this sprint",
  });
  if (description === undefined) {
    return;
  }

  const dueOn = await vscode.window.showInputBox({
    title: "Git Review: Create Milestone — Due Date (optional)",
    prompt: "Enter an optional due date (YYYY-MM-DD), or leave blank.",
    placeHolder: "2025-12-31",
    validateInput: (value: string) => {
      if (!value) {
        return null;
      }
      if (!/^\d{4}-\d{2}-\d{2}$/.test(value.trim())) {
        return "Must be in YYYY-MM-DD format.";
      }
      return null;
    },
  });
  if (dueOn === undefined) {
    return;
  }

  const args: string[] = [
    "git-review",
    "create-milestone",
    `--repo ${shellQuote(repo)}`,
    `--title ${shellQuote(title.trim())}`,
  ];

  if (description.trim()) {
    args.push(`--description ${shellQuote(description.trim())}`);
  }

  if (dueOn.trim()) {
    args.push(`--due-on ${shellQuote(dueOn.trim())}`);
  }

  if (cfg.githubToken) {
    args.push(`--token ${shellQuote(cfg.githubToken)}`);
  }

  runInTerminal(args.join(" "));
}

// ---------------------------------------------------------------------------
// Command: Agile Plan
// ---------------------------------------------------------------------------

async function agilePlan(): Promise<void> {
  const cfg = getConfig();

  if (!cfg.openaiApiKey && !cfg.openaiBaseUrl) {
    const action = await vscode.window.showWarningMessage(
      "No OpenAI API key configured. Open settings to add one.",
      "Open Settings"
    );
    if (action === "Open Settings") {
      await vscode.commands.executeCommand(
        "workbench.action.openSettings",
        "gitReview.openaiApiKey"
      );
    }
    return;
  }

  let repo = cfg.defaultRepo.trim();
  let owner = cfg.defaultOwner.trim();

  if (!repo && !owner) {
    const input = await vscode.window.showInputBox({
      title: "Git Review: Agile Plan",
      prompt:
        "Enter the GitHub repository (owner/repo), owner/org for all repos, or leave blank to enter owner separately.",
      placeHolder: "owner/repo",
      validateInput: (value: string) => {
        if (!value) {
          return null;
        }
        if (!/^[^/]+(\/.+)?$/.test(value.trim())) {
          return "Must be 'owner/repo', 'owner/*', or a bare owner name.";
        }
        return null;
      },
    });
    if (input === undefined) {
      return;
    }
    repo = input.trim();

    if (!repo) {
      const ownerInput = await vscode.window.showInputBox({
        title: "Git Review: Agile Plan — Owner",
        prompt: "Enter the GitHub user or organization name to plan all repos.",
        placeHolder: "my-org",
        validateInput: (value: string) => {
          if (!value || !value.trim()) {
            return "Owner is required.";
          }
          return null;
        },
      });
      if (!ownerInput || !ownerInput.trim()) {
        return;
      }
      owner = ownerInput.trim();
    }
  }

  const args: string[] = ["git-review", "agile"];

  if (repo) {
    args.push(`--repo ${shellQuote(repo)}`);
  } else {
    args.push(`--owner ${shellQuote(owner)}`);
  }

  args.push(`--sprint-capacity ${cfg.sprintCapacity}`);
  args.push(`--sprints ${cfg.sprints}`);
  args.push(...buildCommonArgs(cfg));

  runInTerminal(args.join(" "));
}

// ---------------------------------------------------------------------------
// Extension lifecycle
// ---------------------------------------------------------------------------

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("gitReview.review", reviewRepository),
    vscode.commands.registerCommand("gitReview.commitMessage", generateCommitMessage),
    vscode.commands.registerCommand("gitReview.createIssues", createIssues),
    vscode.commands.registerCommand("gitReview.createMilestone", createMilestone),
    vscode.commands.registerCommand("gitReview.agile", agilePlan)
  );
}

export function deactivate(): void {
  // Nothing to clean up.
}
