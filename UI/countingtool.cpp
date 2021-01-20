#include "countingtool.h"
#include "ui_countingtool.h"

CountingTool::CountingTool(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::CountingTool)
{
    ui->setupUi(this);
}

CountingTool::~CountingTool()
{
    delete ui;
}
