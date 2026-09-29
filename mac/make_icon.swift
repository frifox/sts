// Draws the app icon - a caption bubble: two Russian lines over two English
// ones - into an .iconset (build.sh turns it into AppIcon.icns).
// Usage: swift make_icon.swift <out.iconset>
import AppKit

func color(_ hex: UInt32) -> CGColor {
    CGColor(red: CGFloat(hex >> 16 & 0xFF) / 255, green: CGFloat(hex >> 8 & 0xFF) / 255,
            blue: CGFloat(hex & 0xFF) / 255, alpha: 1)
}

/// The icon on Apple's 1024 grid (an 824 tile, inset 100), drawn at `size`.
func render(_ size: Int) -> Data {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size, bitsPerSample: 8,
                               samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                               colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    let context = NSGraphicsContext(bitmapImageRep: rep)!
    let cg = context.cgContext
    let scale = CGFloat(size) / 1024
    cg.scaleBy(x: scale, y: -scale)  // top-left origin, like the design
    cg.translateBy(x: 0, y: -1024)

    func rounded(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat, _ r: CGFloat) -> CGPath {
        CGPath(roundedRect: CGRect(x: x, y: y, width: w, height: h), cornerWidth: r, cornerHeight: r, transform: nil)
    }

    // Tile, with the soft drop shadow macOS icons have.
    let tile = rounded(100, 100, 824, 824, 185)
    cg.saveGState()
    // Shadows ignore the transform: device space, y up.
    cg.setShadow(offset: CGSize(width: 0, height: -10 * scale), blur: 24 * scale, color: CGColor(gray: 0, alpha: 0.25))
    cg.addPath(tile)
    cg.setFillColor(color(0xFFF7ED))
    cg.fillPath()
    cg.restoreGState()
    cg.addPath(tile)
    cg.setStrokeColor(color(0xFED7AA))
    cg.setLineWidth(6)
    cg.strokePath()

    // The bubble, centered.
    let bubbleX: CGFloat = 210, bubbleY: CGFloat = 278
    cg.addPath(rounded(bubbleX, bubbleY, 604, 467, 103))
    cg.setFillColor(color(0x1F2937))
    cg.fillPath()

    // Caption lines: Russian (gray), then English (amber).
    let lines: [(y: CGFloat, width: CGFloat, color: UInt32)] = [
        (72, 398, 0x9CA3AF), (155, 261, 0x9CA3AF), (264, 440, 0xF59E0B), (347, 302, 0xF59E0B),
    ]
    for line in lines {
        cg.addPath(rounded(bubbleX + 82, bubbleY + line.y, line.width, 48, 24))
        cg.setFillColor(color(line.color))
        cg.fillPath()
    }
    context.flushGraphics()
    return rep.representation(using: .png, properties: [:])!
}

let out = URL(fileURLWithPath: CommandLine.arguments[1])
try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
for points in [16, 32, 128, 256, 512] {
    try render(points).write(to: out.appendingPathComponent("icon_\(points)x\(points).png"))
    try render(points * 2).write(to: out.appendingPathComponent("icon_\(points)x\(points)@2x.png"))
}
