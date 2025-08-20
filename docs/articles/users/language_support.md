# Language Support

This article is about:

- Multi-language interface support in Hatchling
- Switching between available languages
- Language-specific settings and preferences

## Overview

Hatchling provides multi-language support for the user interface, allowing you to use the application in your preferred language. Language switching happens immediately without requiring a restart.

## Supported Languages

Currently supported languages:

- **English (en)** - Default language with complete coverage
- **French (fr)** - Full translation available

## Changing the Interface Language

### List Available Languages

View all available interface languages:

```bash
settings:language:list
```

This will show:

- Language codes (e.g., `en`, `fr`)
- Native language names (e.g., "English", "Français")
- Translation file status

### Set Interface Language

Change to a specific language:

```bash
settings:language:set fr
```

The interface will immediately switch to the selected language for:

- Command descriptions and help text
- Setting names and descriptions
- Error and status messages
- Category display names

### View Current Language

Check the currently active language:

```bash
settings:get ui:language_code
```

## Language Features

### Automatic Fallback

If a translation is missing for the selected language, the system automatically falls back to English to ensure the interface remains functional.

### Dynamic Switching

Language changes take effect immediately without requiring application restart

### Persistent Setting

Your language choice is automatically saved and will be restored when you restart Hatchling.

## Translation Coverage

### Fully Translated Elements

- Setting category names and descriptions
- Individual setting names, descriptions, and hints
- Command names and help text
- Error and informational messages

### Not Translated

- Internal logs
- Technical identifiers and keys
- Configuration file formats
- API endpoints and URLs
- Code examples and syntax
- File extensions

## Technical Implementation

Translation files are stored in TOML format in the `hatchling/config/languages/` directory. Each language has its own file following the naming convention `{language_code}.toml`.

For technical details about the translation system, see the [i18n Support documentation](../devs/i18n_support.md).

## Contributing New Languages

Interested in adding support for your language? See the [Contributing Languages guide](../devs/contribution_guides/contributing_languages.md) for detailed instructions on creating new translations.

> [!Warning]
> Multi-language support is currently in alpha. The contribution process requires manual testing as automated translation validation is not yet available.

## Command Reference

For detailed command syntax, see the [Chat Commands documentation](chat_commands.md#language-management).

### Quick Reference

| Task | Command |
|------|---------|
| List available languages | `settings:language:list` |
| Set interface language | `settings:language:set {code}` |
| Get current language | `settings:get ui:language_code` |
| Reset to default (English) | `settings:reset ui:language_code` |

## Troubleshooting

### Language Not Displaying Correctly

1. Verify the language code is correct: `settings:language:list`
2. Check current setting: `settings:get ui:language_code`
3. Try resetting to English: `settings:language:set en`

### Missing Translations

If you notice untranslated text:

1. Check if the language file is complete
2. Report missing translations at <https://github.com/CrackingShells/Hatchling/issues> to maintainers using the following:

    ```markdown
    **Title:** Missing translation [language-name]

    **Language code:** (e.g., fr)

    **Translation File:**
    Retrieve the translation file associate to the language code at `/path/to/Hatchling/hatchling/config/languages` 
    - [ ] Translation file added to this issue

    **Location in UI:** (Describe where you saw the untranslated text, e.g., command help, error message, setting name)

    **Untranslated text:** (Paste the screenshot of untranslated string)

    **Expected translation:** (If you know the correct translation, provide it here)

    **Additional context:** (Optional: screenshots, steps to reproduce, etc.)
    ```

3. Consider contributing improvements to existing translations yourself 🗺️🙏
