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
    /// Whether new output scrolls into view. Changed only by the user
    /// scrolling (up: stop, back to the bottom: follow again) and by the
    /// console being shown (follow) - not worked out from the scroll position
    /// when output arrives: while the panel is collapsed it has no height, and
    /// that position said "not at the bottom", so it popped up at the top.
    private var followsTail = true
    private var scrollingItself = false
    private var lastClipSize = NSSize.zero

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
        scrollView.contentView.postsBoundsChangedNotifications = true
        NotificationCenter.default.addObserver(
            self, selector: #selector(clipViewMoved),
            name: NSView.boundsDidChangeNotification, object: scrollView.contentView
        )
    }

    /// Shown (or just started): the latest output, following from here on.
    func showTail() {
        followsTail = true
        needsLayout = true
        scrollToEnd()
    }

    override func layout() {
        super.layout()
        if followsTail { scrollToEnd() }  // resized, or the panel opening: keep the tail in view
    }

    private func scrollToEnd() {
        scrollingItself = true
        textView.scrollToEndOfDocument(nil)
        scrollingItself = false
    }

    /// The visible part moved: a scroll by the user decides whether to follow.
    /// Moves that come with a size change are the panel opening or resizing.
    @objc private func clipViewMoved() {
        let size = scrollView.contentView.bounds.size
        defer { lastClipSize = size }
        guard !scrollingItself, size == lastClipSize, size.height > 1 else { return }
        followsTail = isScrolledToBottom
    }

    required init?(coder: NSCoder) { fatalError() }

    func clear() {
        textView.textStorage?.setAttributedString(NSAttributedString())
    }

    func append(_ chunk: String) {
        guard let storage = textView.textStorage else { return }
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
        if followsTail { scrollToEnd() }
    }

    private var isScrolledToBottom: Bool {
        let visible = scrollView.contentView.documentVisibleRect
        return visible.maxY >= textView.bounds.maxY - 20
    }
}
