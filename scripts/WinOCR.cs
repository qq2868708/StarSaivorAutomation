// Windows OCR 工具 — 使用 Windows 10/11 内置 OCR 引擎
// 编译: csc /out:WinOCR.exe scripts/WinOCR.cs
// 用法: WinOCR.exe <image_path>
using System;
using System.IO;
using System.Threading.Tasks;
using Windows.Media.Ocr;
using Windows.Graphics.Imaging;
using Windows.Storage.Streams;

class WinOCR
{
    [STAThread]
    static void Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.Error.WriteLine("用法: WinOCR.exe <image_path>");
            Environment.Exit(1);
        }
        string imagePath = args[0];
        if (!File.Exists(imagePath))
        {
            Console.Error.WriteLine("FILE_NOT_FOUND: " + imagePath);
            Environment.Exit(1);
        }

        try
        {
            var result = RecognizeAsync(imagePath).GetAwaiter().GetResult();
            foreach (var line in result.Lines)
            {
                string text = "";
                foreach (var word in line.Words)
                    text += word.Text;
                Console.WriteLine(text);
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("OCR_FAILED: " + ex.Message);
            Environment.Exit(1);
        }
    }

    static async Task<OcrResult> RecognizeAsync(string imagePath)
    {
        var engine = OcrEngine.TryCreateFromUserProfileLanguages();
        if (engine == null)
            throw new Exception("无法创建 OCR 引擎");

        var file = await Windows.Storage.StorageFile.GetFileFromPathAsync(imagePath);
        using (var stream = await file.OpenReadAsync())
        {
            var decoder = await BitmapDecoder.CreateAsync(stream);
            var softwareBitmap = await decoder.GetSoftwareBitmapAsync();
            return await engine.RecognizeAsync(softwareBitmap);
        }
    }
}
