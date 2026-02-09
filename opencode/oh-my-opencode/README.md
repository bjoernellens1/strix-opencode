# oh-my-opencode

This directory is a placeholder for the [oh-my-opencode](https://github.com/code-yeongyu/oh-my-opencode) plugin.

## Setup as Git Submodule

To add your fork as a submodule, run from the repository root:

```bash
# Remove this placeholder directory first
rm -rf opencode/oh-my-opencode

# Add your fork as a submodule
git submodule add https://github.com/bjoernellens1/oh-my-opencode opencode/oh-my-opencode

# Commit the submodule reference
git add .gitmodules opencode/oh-my-opencode
git commit -m "Add oh-my-opencode as submodule"
```

## Using the Upstream Repo Directly

If you don't have a fork yet:

```bash
rm -rf opencode/oh-my-opencode
git submodule add https://github.com/code-yeongyu/oh-my-opencode opencode/oh-my-opencode
```

## Updating the Submodule

```bash
cd opencode/oh-my-opencode
git pull origin dev
cd ../..
git add opencode/oh-my-opencode
git commit -m "Update oh-my-opencode submodule"
```
