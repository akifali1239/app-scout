// Dumps NLEmbedding (macOS built-in) word vectors. Usage: embed_bin words.txt out.tsv
// Output line: word \t v1,v2,...   (words without a vector are skipped)
import Foundation
import NaturalLanguage

let args = CommandLine.arguments
let words = try! String(contentsOfFile: args[1], encoding: .utf8).split(separator: "\n")
let e = NLEmbedding.wordEmbedding(for: .english)!
var out = ""
for w in words {
    if let v = e.vector(for: String(w)) {
        out += w + "\t" + v.map { String(format: "%.4f", $0) }.joined(separator: ",") + "\n"
    }
}
try! out.write(toFile: args[2], atomically: true, encoding: .utf8)
