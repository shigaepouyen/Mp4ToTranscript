// Renders the macOS app icon (a page with a waveform) on Apple's 1024 px icon grid:
// 824 px body at 100 px, continuous corners. Usage: swift scripts/render_icon.swift out.png
// Same geometry as the layers in macos/AppIcon.icon (Icon Composer source).
import AppKit

let size: CGFloat = 1024
let output = CommandLine.arguments.dropFirst().first ?? "AppIcon-1024.png"

guard let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(size), pixelsHigh: Int(size),
                                 bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                 colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0),
      let context = NSGraphicsContext(bitmapImageRep: rep) else { fatalError("No bitmap context") }
NSGraphicsContext.current = context
let cg = context.cgContext
// Top-left origin, like the SVG layers.
cg.translateBy(x: 0, y: size)
cg.scaleBy(x: 1, y: -1)

func color(_ hex: UInt32, _ alpha: CGFloat = 1) -> CGColor {
    CGColor(srgbRed: CGFloat(hex >> 16 & 0xFF) / 255, green: CGFloat(hex >> 8 & 0xFF) / 255,
            blue: CGFloat(hex & 0xFF) / 255, alpha: alpha)
}

/// Continuous-corner rectangle (squircle-like), as used by macOS icons.
func continuous(_ rect: CGRect, radius: CGFloat) -> CGPath {
    NSBezierPath(roundedRect: rect, xRadius: radius, yRadius: radius).cgPath
}

// Body: drop shadow, then a vertical green gradient.
let body = CGRect(x: 100, y: 100, width: 824, height: 824)
cg.saveGState()
cg.setShadow(offset: CGSize(width: 0, height: 10), blur: 28, color: color(0x000000, 0.3))
cg.addPath(continuous(body, radius: 185))
cg.setFillColor(color(0x3E5528))
cg.fillPath()
cg.restoreGState()

cg.saveGState()
cg.addPath(continuous(body, radius: 185))
cg.clip()
let gradient = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB),
                          colors: [color(0x7A955A), color(0x2E4420)] as CFArray, locations: [0, 1])!
cg.drawLinearGradient(gradient, start: CGPoint(x: 512, y: 100), end: CGPoint(x: 512, y: 924), options: [])
// Soft top highlight.
let shine = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB),
                       colors: [color(0xFFFFFF, 0.18), color(0xFFFFFF, 0)] as CFArray, locations: [0, 1])!
cg.drawLinearGradient(shine, start: CGPoint(x: 512, y: 100), end: CGPoint(x: 512, y: 420), options: [])
cg.restoreGState()

// Page.
let page = CGRect(x: 287, y: 206, width: 450, height: 590)
cg.saveGState()
cg.setShadow(offset: CGSize(width: 0, height: 14), blur: 30, color: color(0x10200A, 0.35))
cg.addPath(continuous(page, radius: 52))
cg.setFillColor(color(0xFAFBF6))
cg.fillPath()
cg.restoreGState()

// Waveform.
cg.setLineCap(.round)
cg.setLineWidth(34)
cg.setStrokeColor(color(0x4C6634))
let bars: [(x: CGFloat, height: CGFloat)] = [(357, 56), (419, 150), (481, 236), (543, 168), (605, 104), (667, 48)]
for bar in bars {
    cg.move(to: CGPoint(x: bar.x, y: 400 - bar.height / 2))
    cg.addLine(to: CGPoint(x: bar.x, y: 400 + bar.height / 2))
}
cg.strokePath()

// Text lines.
cg.setLineWidth(26)
cg.setStrokeColor(color(0x8C9C7A))
for (y, end) in [(588.0, 667.0), (650.0, 667.0), (712.0, 540.0)] as [(CGFloat, CGFloat)] {
    cg.move(to: CGPoint(x: 357, y: y))
    cg.addLine(to: CGPoint(x: end, y: y))
}
cg.strokePath()

NSGraphicsContext.current = nil
guard let png = rep.representation(using: .png, properties: [:]) else { fatalError("PNG encoding failed") }
try png.write(to: URL(filePath: output))
print(output)
