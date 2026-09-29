import AppKit

/// sts's stdout/stderr, IDE style: monospaced, follows the tail unless
/// scrolled up, carriage-return progress bars redraw in place.
final class ConsoleView: NSView {
    private let scrollView = NSScrollView()
    private let textView = NSTextView()
    private let maxCharacters = 1_000_000
    private let attributes: [NSAttributedString.Key: Any] = [
        .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular),
        .foregroundColor: NSColor.textColor,
    ]
    // ESC [ ... letter (colors, cursor moves): shown as plain text.
    private let ansi = try! NSRegularExpression(pattern: "\u{1B}\\[[0-9;?]*[A-Za-z]")

    override init(frame: NSRect) {
        super.init(frame: frame)
        textView.isEditable = false
        textView.isRichText = false
        textView.drawsBackground = true
        textView.backgroundColor = .textBackgroundColor
        textView.textContainerInset = NSSize(width: 6, height: 4)
        textView.autoresizingMask = [.width]
        textView.isVerticallyResizable = true
        textView.textContainer?.widthTracksTextView = true
        scrollView.documentView = textView
        scrollView.hasVerticalScroller = true
        scrollView.autoresizingMask = [.width, .height]
        scrollView.frame = bounds
        addSubview(scrollView)
    }

    required init?(coder: NSCoder) { fatalError() }

    func clear() {
        textView.textStorage?.setAttributedString(NSAttributedString())
    }

    func append(_ chunk: String) {
        guard let storage = textView.textStorage else { return }
        let followTail = isScrolledToBottom
        var text = chunk.replacingOccurrences(of: "\r\n", with: "\n")
        text = ansi.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "")

        storage.beginEditing()
        for (i, part) in text.components(separatedBy: "\r").enumerated() {
            if i > 0 {  // a lone \r: overwrite the current line
                let newline = (storage.string as NSString).range(of: "\n", options: .backwards)
                let lineStart = newline.location == NSNotFound ? 0 : newline.location + 1
                storage.deleteCharacters(in: NSRange(location: lineStart, length: storage.length - lineStart))
            }
            storage.append(NSAttributedString(string: part, attributes: attributes))
        }
        if storage.length > maxCharacters {
            storage.deleteCharacters(in: NSRange(location: 0, length: storage.length - maxCharacters))
        }
        storage.endEditing()
        if followTail { textView.scrollToEndOfDocument(nil) }
    }

    private var isScrolledToBottom: Bool {
        let visible = scrollView.contentView.documentVisibleRect
        return visible.maxY >= textView.bounds.maxY - 20
    }
}
