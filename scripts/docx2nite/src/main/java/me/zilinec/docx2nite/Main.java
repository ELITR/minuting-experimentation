package me.zilinec.docx2nite;

import org.docx4j.openpackaging.exceptions.Docx4JException;
import org.docx4j.openpackaging.packages.WordprocessingMLPackage;
import org.docx4j.openpackaging.parts.WordprocessingML.MainDocumentPart;
import org.docx4j.wml.P;
import org.docx4j.wml.PPrBase;

import java.io.*;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class Main {

    public static WordprocessingMLPackage openDocument(String filename) {
        try {
            return WordprocessingMLPackage.load(new File(filename));
        } catch (Docx4JException e) {
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }

    public static List<String> extractBulletLists(WordprocessingMLPackage doc) {
        List<String> resultLines = new ArrayList<>();
        MainDocumentPart part = doc.getMainDocumentPart();
        List<Object> contents = part.getContent();
        for (Object object: contents) {
            if (object instanceof P) {
                P p = (P) object;
                PPrBase.NumPr numPr = p.getPPr().getNumPr();
                int depth = 0;
                if (numPr != null) {
                    depth = numPr.getIlvl().getVal().intValue() + 1;
//                    numId = numPr.getNumId().getVal();
                }
                StringWriter writer = new StringWriter();
                for (int i = 0; i < depth; i++) {
                    writer.append("-");
                }
                writer.append(p.toString());
                resultLines.add(writer.toString());
            } else{
                System.out.println("Skipping line:");
            }
            System.out.print(object.getClass().getCanonicalName() + "\t");
            System.out.println(object.toString());
        }
        return resultLines;
    }

    public static void writeToXML(String to_filename, List<String> contents, String prefix) {
        // split the directory name from the prefix, so it isn't included in nite ids
        int dirPos = prefix.lastIndexOf("/");
        if (dirPos < prefix.length() - 1) {
            prefix = prefix.substring(dirPos + 1);
        }

        BufferedWriter writer = null;
        try {
            writer = new BufferedWriter(new FileWriter(to_filename));
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        
        try {
            writer.write("<?xml version=\"1.0\"  encoding=\"ISO-8859-1\"?>\n");
            writer.write("<nite:root xmlns:nite=\"http://nite.sourceforge.net/\">\n");
            writer.write("<abstract nite:id=\"" + prefix + "\">\n");
            for (int i = 0; i < contents.size(); i++) {
                String text = contents.get(i);
                String entry = "<sentence nite:id=\"" + prefix + ".s." + i + "\">" + text + "</sentence>\n";
                writer.write(entry);
            }
            writer.write("</abstract>\n");
            writer.write("</nite:root>\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static String guessPrefix(String filename) {
        String prefix = filename;
        int pos = filename.lastIndexOf(".");
        if (pos > 0) {
            prefix = filename.substring(0, pos);
        }
        return prefix;
    }

    public static void main(String[] args) {
        if (args.length < 1 || args[0] == null) {
            System.out.println("Please provide a filename");
            System.exit(1);
        }
        WordprocessingMLPackage doc = openDocument(args[0]);
        List<String> contents = extractBulletLists(doc);
        String prefix = guessPrefix(args[0]);
        writeToXML(prefix + ".annotation.xml", contents, prefix);
        System.out.println("Done!");
    }
}
