# 更改文件名 “猥琐萌**” 至 “***”
$path = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent #获取当前目录
$directory = $path + "\rename"
$files = Get-ChildItem -Path $path -File *.png  # 获取当前文件夹下所有 png 格式的文件
foreach ($file in $files) {
    $fileName = $file.Name[3..($file.Name.length-1)]  # 去掉文件中“猥琐萌”三个字

    $convertNameType = [System.String]$fileName -replace ' ', ''  # System.String 强制转化 System.Object[] to System.String

    $file.FullName
    $fileName.ToString()
    $fileName.GetType()  # 得到变量类型

    Rename-Item -Path $file.FullName -NewName $convertNameType
}