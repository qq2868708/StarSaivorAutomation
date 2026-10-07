// Windows OCR 工具 — 使用 Windows 10/11 内置 OCR 引擎

using Windows.Media.Ocr;
using Windows.Graphics.Imaging;

if (args.Length < 1)
{
    Console.Error.WriteLine("用法: winocr <image_path>");
    return 1;
}

string imagePath = Path.GetFullPath(args[0]);
if (!File.Exists(imagePath))
{
    Console.Error.WriteLine($"FILE_NOT_FOUND: {imagePath}");
    return 1;
}

try
{
    var engine = OcrEngine.TryCreateFromUserProfileLanguages();
    if (engine == null)
    {
        Console.Error.WriteLine("OCR 引擎不可用");
        return 1;
    }

    // 使用 .NET Stream → WinRT IRandomAccessStream (避免 StorageFile 路径问题)
    using var fileStream = File.OpenRead(imagePath);
    using var winrtStream = fileStream.AsRandomAccessStream();
    var decoder = await BitmapDecoder.CreateAsync(winrtStream);
    var softwareBitmap = await decoder.GetSoftwareBitmapAsync();
    var result = await engine.RecognizeAsync(softwareBitmap);

    foreach (var line in result.Lines)
    {
        var words = string.Join("", line.Words.Select(w => w.Text));
        Console.WriteLine(words);
    }

    return 0;
}
catch (Exception ex)
{
    Console.Error.WriteLine($"OCR_FAILED: {ex.Message}");
    return 1;
}
