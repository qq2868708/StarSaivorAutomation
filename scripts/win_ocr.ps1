# Windows OCR — 使用 Windows 10/11 内置 OCR 引擎识别图片文字
# 用法: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/win_ocr.ps1 <image_path>
param([string]$ImagePath)

if (-not (Test-Path $ImagePath)) {
    Write-Error "FILE_NOT_FOUND: $ImagePath"
    exit 1
}

Add-Type -AssemblyName System.Runtime.WindowsRuntime
Add-Type -AssemblyName System.Drawing

# WinRT async → Task 转换
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1
})[0]

try {
    $file = Get-Item $ImagePath
    $stream = [System.IO.File]::OpenRead($file.FullName)

    # 1. 解码图片
    $decoderAsync = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]::CreateAsync($stream)
    $decoderTask = $asTask.MakeGenericMethod([Windows.Graphics.Imaging.BitmapDecoder]).Invoke($null, @($decoderAsync))
    $decoderTask.Wait()
    $decoder = $decoderTask.Result

    # 2. 获取像素
    $pixelAsync = $decoder.GetPixelDataAsync()
    $pixelTask = $asTask.MakeGenericMethod([Windows.Graphics.Imaging.PixelDataProvider]).Invoke($null, @($pixelAsync))
    $pixelTask.Wait()
    $pixels = $pixelTask.Result.DetachPixelData()

    # 3. 创建 SoftwareBitmap
    $sb = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType=WindowsRuntime]::CreateCopyFromBuffer(
        [Windows.Storage.Streams.Buffer, Windows.Storage.Streams, ContentType=WindowsRuntime]::CreateCopyFromMemoryBuffer(
            [Windows.Storage.Streams.MemoryBuffer, Windows.Storage.Streams, ContentType=WindowsRuntime]::CreateFromBuffer(
                [Windows.Storage.Streams.Buffer, Windows.Storage.Streams, ContentType=WindowsRuntime]::CreateCopyFromMemoryBuffer(
                    [Windows.Storage.Streams.MemoryBuffer, Windows.Storage.Streams, ContentType=WindowsRuntime]::CreateFromByteArray($pixels)
                )
            )
        ),
        [Windows.Graphics.Imaging.BitmapPixelFormat]::Rgba8,
        $decoder.PixelWidth,
        $decoder.PixelHeight,
        [Windows.Graphics.Imaging.BitmapAlphaMode]::Premultiplied
    )

    # 4. 创建 OCR 引擎并识别
    $engine = [Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType=WindowsRuntime]::TryCreateFromUserProfileLanguages()
    $ocrAsync = $engine.RecognizeAsync($sb)
    $ocrTask = $asTask.MakeGenericMethod([Windows.Media.Ocr.OcrResult]).Invoke($null, @($ocrAsync))
    $ocrTask.Wait()
    $ocrResult = $ocrTask.Result

    # 5. 输出结果
    foreach ($line in $ocrResult.Lines) {
        $words = ($line.Words | ForEach-Object { $_.Text }) -join ' '
        Write-Output $words
    }

    $stream.Close()
}
catch {
    Write-Error "OCR_FAILED: $_"
    exit 1
}
